from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.model_factory import create_validated_structured_chat_model


FRAMEWORK_PROMPT = """
    你是一名资深学术研究人员，擅长设计结构化、博士研究水平的文献综述框架。

    任务：
    根据给定的研究主题和综述类型，生成一个结构连贯、范围明确的文献综述大纲。
    这是一个文献综述大纲，不是博士研究计划、项目时间表、研究提案、数据库检索方案，也不是系统综述工作流程。
    论文语料库已经由任务预先确定。你无法看到实际可用的论文或文本片段。因此，除非输入中已经明确给出，否则不得假设语料库中存在任何具体的作者、论文、理论、方法、模型、数据集、评价指标、应用场景、时间、历史阶段、研究发现、发展趋势、争议、矛盾、局限性或研究空白。
    不得输出事实性主张、研究结论、引用、参考文献、论文标题，也不得暗示固定论文语料库一定包含支持所有拟定章节的证据。
    不得建议进行外部检索、查找新论文、构造数据库查询、制定纳入与排除标准、设计筛选流程、制定检索策略或执行 PRISMA 流程。

    输出要求：

    1. 仅返回输出结构所要求的字段。
    2. 生成 4 至 7 个章节。
    3. 根据研究主题和综述类型，使用主题、概念、组成部分、关系、应用场景、比较维度或其他分析边界组织大纲。
    4. 除非输入中已经明确给出，否则不得假设领域中存在公认的研究流派、历史阶段、方法类别、学术争议或研究空白。
    5. 除非输入明确要求按照时间演进组织，否则不得将整个框架设计为纯时间顺序结构。
    6. 框架标题、综述范围、章节标题、章节描述和分析维度均使用指定的输出语言。

    7. 每个章节的描述应包含 2 至 4 个句子，并明确界定：
    * 哪些类型的内容属于本章节；
    * 本章节采用的特定分析视角；
    * 本章节与相邻章节之间的语义边界；
    * 本章节如何服务于整篇综述的整体组织。
    章节描述只能界定预期的分析范围，不得陈述研究发现、研究结论、历史事实、已经确认的争议、证据是否存在或语料库覆盖情况。

    8. 每个章节必须包含 2 至 4 条精确的英文 retrieval_hints。
    9. retrieval_hints 仅用于后续针对固定任务论文集构造内部 RAG 查询。它们不是外部检索关键词，也不得用于请求发现额外论文。
    10. section_id 必须使用稳定、唯一的小写 snake_case 格式。

    retrieval_hints 要求：
    1. 每条 retrieval_hint 必须直接锚定给定的研究主题，或者锚定一个简洁且无歧义的主题核心概念。
    2. 每条 retrieval_hint 应由以下两部分组成：
    * 一个主题锚点；
    * 一个当前章节特定的概念、组成部分、关系、目标、属性、比较维度或评价视角。

    3. 使用简洁的英文名词短语或关键词短语。
    4. 不得使用疑问句、命令句、完整句子、布尔检索语法、数据库名称、搜索引擎名称或论文发现指令。
    5. 除非输入中已经明确给出，否则不得引入作者、论文标题、理论、方法、模型、数据集、评价指标、应用场景、时间、研究发现、发展趋势、争议、矛盾、局限性或研究空白。
    6. 避免使用以下缺少限定的通用检索提示：
    * background
    * methods
    * performance
    * advantages
    * challenges
    * limitations
    * future work
    * research gaps

    只有在这些词语同时受到主题锚点和章节特定检索边界限定时，才可以使用。
    7. 每条 retrieval_hint 必须严格位于对应章节 description 所定义的范围内。
    8. 尽量减少不同章节之间重复或近似重复的 retrieval_hints，使不同章节能够从同一语料库中检索出具有区分度的文本片段子集。
    9. 优先使用可能出现在多篇论文中的概念级表达，避免使用可能只出现在单篇论文中的高度具体措辞。
    10. retrieval_hint 可以使用简洁且无歧义的主题锚点，不必逐字重复完整的研究主题。

    推荐的组织方式：
    * 开篇章节可以用于界定研究主题、概念边界、组织问题和综述范围，但不得断言具体历史事实或研究动机。
    * 中间章节应代表彼此区分的主题边界或分析边界。章节数量和组织形式应由研究主题和综述类型决定。
    * 仅当独立的方法、技术或证据设计章节适合当前综述类型，并且能够在不假设特定方法已经存在于语料库中的前提下进行设计时，才应包含该章节。
    * 最后的综合章节可以为后续基于证据分析局限性、结果不一致、未解决问题或未来启示预留空间。在框架生成阶段，该章节只能界定需要检查的证据类别，不得虚构具体研究空白、矛盾或未来研究方向。

    综述类型指导：
    * narrative：
    强调主题综合、概念关系和连贯的结构组织。只有在输入明确要求，或者后续检索证据能够支持时，才讨论概念演进。
    * critical：
    为理论、方法和证据层面的评价设置清晰的分析空间。将批判性分析视为一种分析维度，而不是预先确定一组研究缺陷。
    * scoping：
    强调对固定语料库中所包含的主题、概念、研究对象、应用场景或证据类别进行广泛映射。不得声称执行了正式检索、筛选流程或 PRISMA 流程。
    * systematic：
    强调对固定语料库中已有证据进行结构化、可复现的组织。不得声称执行了数据库检索、筛选协议、纳入与排除流程或 PRISMA 工作流程。
""".strip()



class FrameworkSection(BaseModel):
    section_id: str = Field(
        pattern = r"^[a-z][a-z0-9_]*$",
        min_length = 2,
        max_length = 64,
    )
    title: str = Field(min_length = 2, max_length = 120)
    description: str = Field(min_length = 10, max_length = 500)
    retrieval_hints: list[str] = Field(
        min_length = 2,
        max_length = 4,
    )


class ReviewFramework(BaseModel):
    title: str = Field(min_length = 2, max_length = 200)
    scope: str = Field(min_length = 10, max_length = 600)
    sections: list[FrameworkSection] = Field(
        min_length = 4,
        max_length = 7,
    )

    @model_validator(mode = "after")
    def section_ids_must_be_unique(self) -> "ReviewFramework":
        section_ids = [
            section.section_id
            for section in self.sections
        ]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("section_id must be unique")
        return self


class GenerateFrameworkNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        
        self.model = model or create_validated_structured_chat_model(
            ReviewFramework,
            temperature = 0,
        )

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        if state.get("framework_artifact_ref"):
            return {
                "stage": "generating_claims",
                "status": "running",
            }

        model_input = {
            "topic": state["topic"],
            "output_language": state["language"],
            "review_type": state["review_type"],
        }

        try:
            result = await self.model.ainvoke(
                [
                    SystemMessage(content = FRAMEWORK_PROMPT),
                    HumanMessage(
                        content = json.dumps(
                            model_input,
                            ensure_ascii = False,
                        )
                    ),
                ]
            )

            framework = (
                result
                if isinstance(result, ReviewFramework)
                else ReviewFramework.model_validate(result)
            )

            framework_data = framework.model_dump(mode = "json")
            framework_hash = hashlib.sha256(
                json.dumps(
                    framework_data,
                    ensure_ascii = False,
                    sort_keys = True,
                    separators = (",", ":"),
                ).encode("utf-8")
            ).hexdigest()

            artifact = await self.artifact_store.write_json(
                run_id = state["run_id"],
                step_key = "generate_framework",
                source = "task_review",
                kind = "task_review_framework_json",
                count = len(framework.sections),
                payload = {
                    "task_id": state["task_id"],
                    "topic": state["topic"],
                    "language": state["language"],
                    "review_type": state["review_type"],
                    "framework_hash": framework_hash,
                    "framework": framework_data,
                },
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"framework generation failed: {exc}",
            }

        return {
            "framework_artifact_ref": artifact.artifact_uri,
            "framework_hash": framework_hash,
            "stage": "generating_claims",
            "status": "running",
            "error": None,
        }
