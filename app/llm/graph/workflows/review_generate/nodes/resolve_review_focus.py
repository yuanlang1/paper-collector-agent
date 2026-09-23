from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ChatClient, ModelOptions
from app.llm.graph.workflows.review_generate.contracts import batches, invoke, save
from app.llm.graph.workflows.review_generate.nodes.finalize import failed


THEME_COMPRESSION_PROMPT = "按主题压缩输入概览或主题摘要。每篇论文必须且只能归入一个主题，保持全部论文ID。" "摘要总长度须短于输入；概览和摘要都不是正文证据。"

REVIEW_FOCUS_PROMPT = (
    "阅读后细化综述问题、范围和标题。可去掉预设结论，不可改变用户核心研究对象。"
    "明显语料不匹配标记 corpus_scope_mismatch。区分核心与补充问题，记录无法回答的问题。"
    "问题只表示待验证方向，概览不证明结论。使用输出语言，保持固定论文集；"
    "未映射到研究问题的论文须在 excluded_papers 说明其不相关原因。"
)


class Question(BaseModel):
    question: str = Field(min_length=1)
    core: bool
    relevant_paper_ids: list[str]


class ExcludedPaper(BaseModel):
    paper_id: str
    reason: str = Field(min_length=1)


class ReviewFocus(BaseModel):
    title: str = Field(min_length=2)
    scope: str = Field(min_length=10)
    adjustment_reason: str
    corpus_scope_mismatch: bool
    research_questions: list[Question] = Field(min_length=1)
    unanswerable_questions: list[str]
    excluded_papers: list[ExcludedPaper] = Field(default_factory=list)


class Theme(BaseModel):
    summary: str = Field(max_length=800)
    paper_ids: list[str] = Field(min_length=1)


class Themes(BaseModel):
    themes: list[Theme]


class ResolveReviewFocusNode:
    def __init__(self, *, artifact_store=None, model=None, theme_model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.chat = chat or ChatClient()
        self.model = model or self.chat.structured(ReviewFocus, options=ModelOptions(temperature=0))
        self.theme_model = theme_model

    async def __call__(self, state):
        try:
            studies = (
                await self.artifact_store.read_json_uri(state["study_records_artifact_ref"])
            )["studies"]
            overview = studies
            groups = list(batches(overview))
            while len(groups) > 1:
                model = self.theme_model or self.chat.structured(
                    Themes, options=ModelOptions(temperature=0)
                )
                compact = []
                for group in groups:
                    result = await invoke(
                        model, Themes, THEME_COMPRESSION_PROMPT, {"records": group},
                    )
                    expected = [
                        pid
                        for record in group
                        for pid in (
                            record["paper_ids"] if "paper_ids" in record else [record["paper_id"]]
                        )
                    ]
                    ids = [pid for theme in result["themes"] for pid in theme["paper_ids"]]
                    if sorted(ids) != sorted(expected):
                        raise ValueError("theme mapping must cover every paper exactly once")
                    compact.extend(result["themes"])
                next_groups = list(batches(compact))
                if len(next_groups) >= len(groups):
                    raise ValueError("theme compression did not reduce context; narrow the corpus")
                overview, groups = compact, next_groups
            focus = await invoke(
                self.model,
                ReviewFocus,
                REVIEW_FOCUS_PROMPT,
                {
                    "topic": state["topic"],
                    "output_language": state["language"],
                    "review_type": state["review_type"],
                    "overview": overview,
                },
            )
            for index, question in enumerate(focus["research_questions"], 1):
                question["question_id"] = f"rq_{index:03d}"
                if not set(question["relevant_paper_ids"]) <= set(state["paper_ids_snapshot"]):
                    raise ValueError("focus references papers outside task")
            mapped = {
                pid
                for question in focus["research_questions"]
                for pid in question["relevant_paper_ids"]
            }
            excluded = {paper["paper_id"] for paper in focus["excluded_papers"]}
            if not focus["corpus_scope_mismatch"] and (
                mapped | excluded != set(state["paper_ids_snapshot"]) or mapped & excluded
            ):
                raise ValueError("each paper must be mapped to a question or explicitly excluded")
            if not focus["corpus_scope_mismatch"] and not any(
                question["core"] for question in focus["research_questions"]
            ):
                raise ValueError("review needs at least one core research question")
            ref = await save(
                self.artifact_store, state, "review_focus", {"focus": focus, "overview": overview}
            )
            update = {"review_focus": focus, "review_focus_artifact_ref": ref}
            if focus["corpus_scope_mismatch"]:
                return failed(
                    focus["adjustment_reason"], **update, error_code="CORPUS_SCOPE_MISMATCH"
                )
            return {**update, "stage": "generating_claims", "status": "running"}
        except Exception as exc:
            return failed(
                f"focus resolution failed: {exc}", error_code="FOCUS_FAILED", retryable=True
            )
