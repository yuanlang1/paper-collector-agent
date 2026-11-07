from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    identity: list[str] = Field(default_factory=list)
    research_interests: list[str] = Field(default_factory=list)
    communication: list[str] = Field(default_factory=list)
    long_term_goals: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    subject: str
    content: str
    context: str | None = None  # 适用条件


class Episode(BaseModel):
    goal: str
    actions: list[str]
    result: str
    lesson: str | None = None
