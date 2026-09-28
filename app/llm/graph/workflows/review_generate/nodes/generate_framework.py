from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    content_hash,
    framework_input_hash,
    invoke,
    read_optional_json,
    save,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


FRAMEWORK_PROMPT = """
你是一名学术综述规划者。依据输入的论文画像规划综述框架；
论文画像只用于组织议题，不是已经核验的原文结论。

章节标题表示讨论对象，description 说明本节覆盖范围，
discussion_questions 列出后续可由论文原文证据回答的比较、条件或局限问题。
章节按主题组织且尽量避免重叠；同一论文可服务于多个章节。每篇输入论文必须归入至少一个章节，
或在 excluded_papers 中给出具体排除理由。保持固定论文集，不得虚构系统综述检索、筛选或质量评价流程。
识别明显的语料与主题不匹配，并使用 output_language。

以下仅为格式示例；使用实际输入中的论文 ID 和输出语言，勿照抄内容：
```json
{
  "title": "主题的研究综述",
  "scope": "比较给定论文中的方法、结果与适用条件。",
  "adjustment_reason": "",
  "corpus_scope_mismatch": false,
  "sections": [
    {
      "section_id": "methods",
      "title": "研究方法与适用条件",
      "description": "比较不同研究设计及其适用范围。",
      "discussion_questions": ["不同方法在何种条件下得到可比结果？"],
      "relevant_paper_ids": ["paper_001", "paper_002"]
    }
  ],
  "excluded_papers": []
}
```
""".strip()


class ExcludedPaper(BaseModel):
    """未纳入任一综述章节的输入论文及其理由。"""

    paper_id: str = Field(description="被排除的当前任务论文 ID。")
    reason: str = Field(min_length=1, description="该论文不适合当前综述范围的具体原因。")


class FrameworkSection(BaseModel):
    """综述 Framework 中一个按主题组织的章节。"""

    section_id: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="稳定的章节标识，只能使用小写字母、数字和下划线，并以字母开头。",
    )
    title: str = Field(
        min_length=1,
        description="面向读者的章节标题，应描述讨论对象而非预设结论。",
    )
    description: str = Field(
        min_length=1,
        description="章节覆盖范围及与其他章节的边界说明。",
    )
    discussion_questions: list[str] = Field(
        min_length=1,
        max_length=4,
        description="本章节需要回答的研究问题，围绕比较、条件或局限展开，不预设未经证据核验的结论。",
    )
    relevant_paper_ids: list[str] = Field(
        min_length=1,
        description="与本章节相关的当前任务论文 ID；同一论文可出现在多个章节。",
    )


class ReviewFramework(BaseModel):
    """由论文画像生成、供 Claim 和章节写作使用的综述框架。"""

    title: str = Field(
        min_length=2,
        description="综述标题，概括主题而不把待核验结论写成事实。",
    )
    scope: str = Field(
        min_length=10,
        description="综述覆盖对象、比较维度和边界的简要说明。",
    )
    adjustment_reason: str = Field(
        default="",
        description="语料与主题不匹配时说明需要调整任务范围的原因；无不匹配时为空。",
    )
    corpus_scope_mismatch: bool = Field(
        default=False,
        description="是否发现当前论文集与综述主题存在明显范围不匹配。",
    )
    sections: list[FrameworkSection] = Field(
        min_length=1,
        description="按主题组织的非空章节列表。",
    )
    excluded_papers: list[ExcludedPaper] = Field(
        default_factory=list,
        description="未映射到任何章节的输入论文及排除理由。",
    )

    @model_validator(mode="after")
    def unique_sections(self):
        ids = [section.section_id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate section IDs")
        return self


class GenerateFrameworkNode:
    def __init__(self, *, artifact_store=None, model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or (chat or ChatClient()).structured(
            ReviewFramework, options=ModelOptions(temperature=0)
        )

    async def __call__(self, state):
        try:
            studies = (
                await self.artifact_store.read_json_uri(state["study_records_artifact_ref"])
            )["studies"]
            old_payload = await read_optional_json(
                self.artifact_store, state.get("framework_artifact_ref")
            )
            input_hash = framework_input_hash(
                state["topic"], state["language"], state["review_type"], studies
            )
            if (
                old_payload
                and old_payload.get("framework_input_hash") == input_hash
            ):
                return {"stage": "generating_claims", "status": "running"}

            framework = await invoke(
                self.model,
                ReviewFramework,
                FRAMEWORK_PROMPT,
                {
                    "topic": state["topic"],
                    "output_language": state["language"],
                    "review_type": state["review_type"],
                    "overview": studies,
                },
            )
            return await self._save_framework(state, old_payload, framework, input_hash)
        except Exception as exc:
            return failed(
                f"framework generation failed: {exc}", error_code="FRAMEWORK_FAILED", retryable=True
            )

    async def _save_framework(self, state, old_payload, framework, input_hash):
        paper_ids = set(state["paper_ids_snapshot"])
        mapped = {
            paper_id
            for section in framework["sections"]
            for paper_id in section["relevant_paper_ids"]
        }
        excluded = {item["paper_id"] for item in framework["excluded_papers"]}
        if not mapped <= paper_ids or not excluded <= paper_ids:
            raise ValueError("framework references papers outside task")
        if not framework["corpus_scope_mismatch"] and (
            mapped | excluded != paper_ids or mapped & excluded
        ):
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
            "stage": "generating_claims",
            "status": "running",
        }
