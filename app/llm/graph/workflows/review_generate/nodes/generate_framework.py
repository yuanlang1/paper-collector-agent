from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    batches,
    content_hash,
    framework_input_hash,
    invoke,
    revisions_for,
    save,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


THEME_COMPRESSION_PROMPT = (
    "按主题压缩输入概览或主题摘要。每篇论文必须且只能归入一个主题，保持全部论文 ID。"
    "摘要总长度须短于输入；概览和摘要都不是正文证据。"
)

FRAMEWORK_PROMPT = (
    "依据论文画像规划综述框架。章节标题应表示讨论对象，不预设未经原文核验的结论。"
    "每节给出需要回答的讨论问题和相关论文 ID；同一论文可以属于多个章节。"
    "未使用的论文必须说明排除原因。保持固定论文集，不得虚构系统综述检索、筛选或质量评价流程。"
    "重规划时，未改变的章节保留原 section_id；新章节使用新 ID。"
    "对每个待处理的旧章节修订，在 section_revision_targets 中给出承接它的新 section_id；"
    "仅当框架层已处理该意见时才使用 null。"
    "识别明显的语料与主题不匹配；使用输出语言。"
)


class ExcludedPaper(BaseModel):
    paper_id: str
    reason: str = Field(min_length=1)


class Theme(BaseModel):
    summary: str = Field(max_length=800)
    paper_ids: list[str] = Field(min_length=1)


class Themes(BaseModel):
    themes: list[Theme]


class FrameworkSection(BaseModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    relevant_paper_ids: list[str] = Field(min_length=1)


class ReviewFramework(BaseModel):
    title: str = Field(min_length=2)
    scope: str = Field(min_length=10)
    adjustment_reason: str = ""
    corpus_scope_mismatch: bool = False
    sections: list[FrameworkSection] = Field(min_length=1)
    excluded_papers: list[ExcludedPaper] = Field(default_factory=list)
    section_revision_targets: dict[str, str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_sections(self):
        ids = [section.section_id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate section IDs")
        return self


class GenerateFrameworkNode:
    def __init__(self, *, artifact_store=None, model=None, theme_model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.chat = chat or ChatClient()
        self.model = model or self.chat.structured(ReviewFramework, options=ModelOptions(temperature=0))
        self.theme_model = theme_model

    async def _overview(self, studies):
        overview = studies
        groups = list(batches(overview))
        while len(groups) > 1:
            model = self.theme_model or self.chat.structured(
                Themes, options=ModelOptions(temperature=0)
            )
            compact = []
            for group in groups:
                result = await invoke(model, Themes, THEME_COMPRESSION_PROMPT, {"records": group})
                expected = [
                    paper_id
                    for record in group
                    for paper_id in record.get("paper_ids", [record["paper_id"]])
                ]
                ids = [paper_id for theme in result["themes"] for paper_id in theme["paper_ids"]]
                if sorted(ids) != sorted(expected):
                    raise ValueError("theme mapping must cover every paper exactly once")
                compact.extend(result["themes"])
            next_groups = list(batches(compact))
            if len(next_groups) >= len(groups):
                raise ValueError("theme compression did not reduce context; narrow the corpus")
            overview, groups = compact, next_groups
        return overview

    async def __call__(self, state):
        try:
            studies = (
                await self.artifact_store.read_json_uri(state["study_records_artifact_ref"])
            )["studies"]
            claims = (
                (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))["claims"]
                if state.get("claims_artifact_ref")
                else []
            )
            verification = (
                (await self.artifact_store.read_json_uri(state["claim_verification_artifact_ref"]))["claims"]
                if state.get("claim_verification_artifact_ref")
                else []
            )
            input_hash = framework_input_hash(
                state["topic"], state["language"], state["review_type"], studies
            )
            old_payload = (
                await self.artifact_store.read_json_uri(state["framework_artifact_ref"])
                if state.get("framework_artifact_ref")
                else None
            )
            revisions = revisions_for(state, "framework")
            section_revisions = [
                item
                for item in state.get("revision_items", [])
                if item["target_type"] in {"section", "add_claim"}
            ]
            planning_revisions = revisions + section_revisions
            if old_payload and old_payload.get("framework_input_hash") == input_hash and not revisions:
                return {"stage": "generating_claims", "status": "running"}

            framework = await invoke(
                self.model,
                ReviewFramework,
                FRAMEWORK_PROMPT,
                {
                    "topic": state["topic"],
                    "output_language": state["language"],
                    "review_type": state["review_type"],
                    "overview": await self._overview(studies),
                    "old_framework": old_payload and old_payload["framework"],
                    "claims": claims,
                    "verification": [
                        {key: item[key] for key in ("claim_id", "status", "reason")}
                        for item in verification
                    ],
                    "revisions": planning_revisions,
                },
            )
            section_revision_targets = framework.pop("section_revision_targets")
            paper_ids = set(state["paper_ids_snapshot"])
            mapped = {
                paper_id
                for section in framework["sections"]
                for paper_id in section["relevant_paper_ids"]
            }
            excluded = {item["paper_id"] for item in framework["excluded_papers"]}
            if not mapped <= paper_ids or not excluded <= paper_ids:
                raise ValueError("framework references papers outside task")
            if not framework["corpus_scope_mismatch"] and (mapped | excluded != paper_ids or mapped & excluded):
                raise ValueError("each paper must be assigned to a section or excluded")
            if framework["corpus_scope_mismatch"]:
                return failed(framework["adjustment_reason"], error_code="CORPUS_SCOPE_MISMATCH")

            old_sections = {
                section["section_id"]: section
                for section in (old_payload or {}).get("framework", {}).get("sections", [])
            }
            changed_sections = [
                section["section_id"]
                for section in framework["sections"]
                if old_sections.get(section["section_id"]) != section
            ]
            section_ids = {section["section_id"] for section in framework["sections"]}
            if revisions and set(section_revision_targets) != {
                item["target_id"] for item in section_revisions
            }:
                raise ValueError("framework must route every pending section revision")
            if any(
                target is not None and target not in section_ids
                for target in section_revision_targets.values()
            ):
                raise ValueError("framework routed a revision to an unknown section")
            stale_claim_ids = {
                claim["claim_id"]
                for claim in claims
                if claim["section_id"] in changed_sections
                or claim["section_id"] not in section_ids
            }
            remaining_revisions = []
            for item in state.get("revision_items", []):
                if item["target_type"] == "framework":
                    continue
                if item["target_type"] in {"claim", "retrieval"} and item["target_id"] in stale_claim_ids:
                    continue
                if item["target_type"] in {"section", "add_claim"}:
                    target = section_revision_targets.get(item["target_id"], item["target_id"])
                    if target is None:
                        continue
                    remaining_revisions.append({**item, "target_id": target})
                else:
                    remaining_revisions.append(item)
            digest = content_hash(framework)
            ref = await save(
                self.artifact_store,
                state,
                "framework",
                {
                    "framework": framework,
                    "framework_hash": digest,
                    "framework_input_hash": input_hash,
                    "changed_section_ids": changed_sections,
                },
            )
            return {
                "framework_artifact_ref": ref,
                "framework_hash": digest,
                "changed_section_ids": changed_sections,
                "revision_items": remaining_revisions,
                "stage": "generating_claims",
                "status": "running",
            }
        except Exception as exc:
            return failed(
                f"framework generation failed: {exc}", error_code="FRAMEWORK_FAILED", retryable=True
            )
