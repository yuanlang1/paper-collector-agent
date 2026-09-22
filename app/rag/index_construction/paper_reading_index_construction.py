from langchain_core.documents import Document
from pydantic import BaseModel, Field, model_validator
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.rag.index_construction.base import BaseQdrantIndexConstructionModule


class ArticleProfile(BaseModel):
    core_problem: str = Field(min_length=1, max_length=800)
    methods: str = Field(min_length=1, max_length=800)
    main_discussion: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def usable(self):
        if all(
            not value.strip() or value.strip() == "not_reported"
            for value in self.model_dump().values()
        ):
            raise ValueError("paper reading has no usable content")
        return self


class PaperReadingIndexConstructionModule(BaseQdrantIndexConstructionModule):
    def __init__(self, **kwargs):
        super().__init__(collection_name=settings.PAPER_READING_COLLECTION_NAME, **kwargs)

    def _get_business_id(self, document):
        return str(document.metadata["paper_id"])

    def _build_embedding_text(self, document):
        return document.page_content

    def _build_payload(self, document):
        return document.metadata

    async def _ensure_collection(self, embedding_dimension):
        try:
            await super()._ensure_collection(embedding_dimension)
        except UnexpectedResponse as exc:
            # Another paper/process may create this lazy collection concurrently.
            if exc.status_code != 409 or not await self.collection_exists():
                raise

    async def lookup(self, paper_id):
        if not await self.collection_exists():
            return None
        records = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._build_point_id(str(paper_id))],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        try:
            if str(payload["paper_id"]) != str(paper_id):
                return None
            ArticleProfile.model_validate(payload["profile"])
        except (ValueError, KeyError):
            return None
        return payload

    @staticmethod
    def payload(paper_id, profile):
        return {"paper_id": str(paper_id), "profile": profile.model_dump()}

    async def save(self, payload):
        profile = ArticleProfile.model_validate(payload["profile"])
        await self.build_index(
            [Document(page_content="\n\n".join(profile.model_dump().values()), metadata=payload,)]
        )
