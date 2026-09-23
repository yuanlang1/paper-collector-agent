from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ChatClient, ModelOptions
from app.llm.graph.workflows.review_generate.contracts import (
    content_hash,
    invoke,
    revisions_for,
    save,
    verified_pack,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed


FRAMEWORK_PROMPT = (
    "依据已核验论点组织综述大纲，不限定节数。每条 Claim 恰好归属一个章节，章节不得为空。"
    "保持 focus 标题和范围，使用输出语言。核心问题不能通过删除章节隐藏。"
    "这是固定论文集的结构化综合，不能虚构系统综述检索、筛选或质量评价流程。"
)


class FrameworkSection(BaseModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    description: str
    claim_ids: list[str] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)


class ReviewFramework(BaseModel):
    title: str
    scope: str
    sections: list[FrameworkSection] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sections(self):
        ids = [section.section_id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate section IDs")
        return self


class GenerateFrameworkNode:
    def __init__(self, *, artifact_store=None, model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or (chat or ChatClient()).structured(ReviewFramework, options=ModelOptions(temperature=0))

    async def __call__(self, state):
        try:
            claims = (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))[
                "claims"
            ]
            verdicts = {
                item["claim_id"]: item
                for item in (
                    await self.artifact_store.read_json_uri(
                        state["claim_verification_artifact_ref"]
                    )
                )["claims"]
            }
            active = [claim for claim in claims if not claim["withdrawn_reason"]]
            for claim in active:
                verified_pack(claim, verdicts[claim["claim_id"]])
            old = None
            if state.get("framework_artifact_ref"):
                old = (await self.artifact_store.read_json_uri(state["framework_artifact_ref"]))[
                    "framework"
                ]
            ids = [claim["claim_id"] for claim in active]
            if (
                old
                and not revisions_for(state, "framework")
                and sorted(ids)
                == sorted(cid for section in old["sections"] for cid in section["claim_ids"])
            ):
                return {"stage": "rendering_sections", "status": "running"}
            framework = await invoke(
                self.model,
                ReviewFramework,
                FRAMEWORK_PROMPT,
                {
                    "focus": state["review_focus"],
                    "claims": active,
                    "old_framework": old,
                    "revisions": revisions_for(state, "framework"),
                    "output_language": state["language"],
                },
            )
            mapped = [cid for section in framework["sections"] for cid in section["claim_ids"]]
            if sorted(mapped) != sorted(ids) or any(
                not section["claim_ids"] for section in framework["sections"]
            ):
                raise ValueError("framework must assign every active claim exactly once")
            framework["title"] = state["review_focus"]["title"]
            framework["scope"] = state["review_focus"]["scope"]
            digest = content_hash(framework)
            ref = await save(
                self.artifact_store,
                state,
                "framework",
                {"framework": framework, "framework_hash": digest},
            )
            return {
                "framework_artifact_ref": ref,
                "framework_hash": digest,
                "stage": "rendering_sections",
                "status": "running",
            }
        except Exception as exc:
            return failed(
                f"framework generation failed: {exc}", error_code="FRAMEWORK_FAILED", retryable=True
            )
