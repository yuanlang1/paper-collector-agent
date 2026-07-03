class OrchestrationRequest(BaseModel):
    topic: str = Field(..., description="研究主题", min_length=2)
    keywords: List[str] = Field(
        ..., description="研究关键词列表", min_length=1
    )
    paper_limit: int = Field(
        default=30, ge=5, le=500,
        description="每节检索的文献数量上限（Scoping Review 可设更高）"
    )
    language: str = Field(
        default="zh-CN",
        description="输出语言: 'zh-CN' 或 'en'"
    )
    citation_style: str = Field(
        default="harvard",
        description="引用格式: 'harvard' (默认), 'apa', 'ieee', 'chicago', 'vancouver'"
    )
    year_from: Optional[int] = Field(default=None, description="起始年份")
    year_to: Optional[int] = Field(default=None, description="结束年份")
    custom_instructions: Optional[str] = Field(
        default=None,
        description="自定义指令（附加到框架生成 prompt 中）"
    )
    use_local_only: bool = Field(
        default=False,
        description="是否仅使用本地已有文献（不进行在线搜索）"
    )
    # Opt-3: 必引文献保底
    must_cite_paper_ids: List[int] = Field(
        default_factory=list,
        description="必须引用的文献 ID 列表，无论检索结果如何都注入每节 context"
    )
    # Opt-9: 综述类型
    review_type: str = Field(
        default="narrative",
        description="综述类型: 'narrative'(叙述性), 'systematic'(系统性), 'scoping'(范围), 'critical'(批判性)"
    )
    # Opt-10: 使用结构化 claims pipeline
    use_claims_pipeline: bool = Field(
        default=False,
        description="是否使用 claims-evidence-render 三阶段管线（更精准但更慢）"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "TOD与文化遗产保护",
                "keywords": ["TOD", "heritage conservation", "sustainable urban design"],
                "paper_limit": 30,
                "language": "en",
                "citation_style": "harvard",
                "year_from": 2015,
            }
        }


