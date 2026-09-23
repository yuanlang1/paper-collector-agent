from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ChatClient, ModelOptions
from app.llm.graph.workflows.review_generate.citations import citation_anchor_ids
from app.llm.graph.workflows.review_generate.contracts import (
    batches,
    content_hash,
    invoke,
    revisions_for,
    save,
    verified_pack,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed


SECTION_DRAFT_PROMPT = (
    "基于当前已核验论点和证据包写综述论证片段，使用 output_language。"
    "以问题组织比较，不逐篇罗列，不扩展范围；保留分歧。"
    "具体事实必须使用对应 [[REF_论文ID]] 引用。"
    "每个片段关联输入中的 claim_ids 和 evidence_ids。重写须执行修订要求。"
)


class Argument(BaseModel):
    text: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class SectionDraft(BaseModel):
    arguments: list[Argument] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=500)


class RenderSectionsNode:
    def __init__(self, *, artifact_store=None, model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or (chat or ChatClient()).structured(SectionDraft, options=ModelOptions(temperature=0))

    async def __call__(self, state):
        refs = dict(state.get("section_draft_artifact_refs", {}))
        try:
            framework = (await self.artifact_store.read_json_uri(state["framework_artifact_ref"]))[
                "framework"
            ]
            claims = {
                claim["claim_id"]: claim
                for claim in (
                    await self.artifact_store.read_json_uri(state["claims_artifact_ref"])
                )["claims"]
            }
            verdicts = {
                item["claim_id"]: item
                for item in (
                    await self.artifact_store.read_json_uri(
                        state["claim_verification_artifact_ref"]
                    )
                )["claims"]
            }
            refs = {
                key: ref
                for key, ref in refs.items()
                if key in {section["section_id"] for section in framework["sections"]}
            }
            previous_summary = ""
            for section in framework["sections"]:
                packages = [
                    {"claim": claims[cid], "evidence": verified_pack(claims[cid], verdicts[cid])}
                    for cid in section["claim_ids"]
                ]
                signature = content_hash(
                    [section, packages, state["language"], state["review_focus"]]
                )
                old = None
                old_ref = state.get("section_draft_artifact_refs", {}).get(section["section_id"])
                if old_ref:
                    old = await self.artifact_store.read_json_uri(old_ref)
                revisions = revisions_for(state, "section", section["section_id"])
                if old and old.get("input_hash") == signature and not revisions:
                    refs[section["section_id"]] = old_ref
                    previous_summary = old["summary"]
                    continue
                arguments, summaries = [], []
                for group in batches(packages, budget=24000):
                    result = await invoke(
                        self.model,
                        SectionDraft,
                        SECTION_DRAFT_PROMPT,
                        {
                            "output_language": state["language"],
                            "focus": state["review_focus"],
                            "framework": framework,
                            "section": section,
                            "claims_with_evidence": group,
                            "old_draft": old,
                            "revisions": revisions,
                            "previous_section_summary": previous_summary,
                        },
                    )
                    group_ids = {item["claim"]["claim_id"] for item in group}
                    evidence = {
                        item["evidence_id"]: item for pack in group for item in pack["evidence"]
                    }
                    for argument in result["arguments"]:
                        if (
                            not set(argument["claim_ids"]) <= group_ids
                            or not set(argument["evidence_ids"]) <= evidence.keys()
                        ):
                            raise ValueError(
                                "section argument references unknown claim or evidence"
                            )
                        papers = {evidence[eid]["paper_id"] for eid in argument["evidence_ids"]}
                        anchors = set(citation_anchor_ids(argument["text"]))
                        if not anchors or not anchors <= papers:
                            raise ValueError("section citations are not bound to supplied evidence")
                        for cid in argument["claim_ids"]:
                            matching = {item["evidence_id"] for item in verdicts[cid]["evidence"]}
                            bound = [
                                evidence[eid] for eid in argument["evidence_ids"] if eid in matching
                            ]
                            support = {
                                item["paper_id"] for item in bound if item["relation"] == "supports"
                            }
                            required = (
                                2
                                if claims[cid]["evidence_requirement"] == "multiple_fulltext"
                                else 1
                            )
                            if len(support & anchors) < required:
                                raise ValueError("argument omitted required claim sources")
                    used = {
                        cid for argument in result["arguments"] for cid in argument["claim_ids"]
                    }
                    if used != group_ids:
                        raise ValueError("section draft omitted verified claims")
                    arguments.extend(result["arguments"])
                    summaries.append(result["summary"])
                draft = {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "framework_hash": state["framework_hash"],
                    "input_hash": signature,
                    "claim_hashes": {
                        cid: claims[cid]["claim_hash"] for cid in section["claim_ids"]
                    },
                    "arguments": arguments,
                    "text": "\n\n".join(item["text"] for item in arguments),
                    "summary": " ".join(summaries),
                    "used_claim_ids": section["claim_ids"],
                    "omitted_claim_ids": [],
                }
                refs[section["section_id"]] = await save(
                    self.artifact_store, state, "section_draft", draft
                )
                previous_summary = draft["summary"]
            return {
                "section_draft_artifact_refs": refs,
                "stage": "assembling_review",
                "status": "running",
            }
        except Exception as exc:
            return failed(
                f"section rendering failed: {exc}",
                section_draft_artifact_refs=refs,
                error_code="RENDER_FAILED",
                retryable=True,
            )
