from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CrossrefSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doi: str | None = Field(None, min_length=1, max_length=500)
    query: str | None = Field(None, min_length=1, max_length=500)
    title: str | None = Field(None, min_length=1, max_length=500)
    author: str | None = Field(None, min_length=1, max_length=200)
    issn: str | None = Field(None, min_length=4, max_length=20)
    from_pub_date: date | None = None
    until_pub_date: date | None = None
    work_type: str | None = Field(None, min_length=1, max_length=80)
    limit: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_request(self) -> "CrossrefSearchArgs":
        search_fields = [
            self.query,
            self.title,
            self.author,
            self.issn,
            self.from_pub_date,
            self.until_pub_date,
            self.work_type,
        ]

        if self.doi and any(value is not None for value in search_fields):
            raise ValueError(
                "doi cannot be combined with Crossref search filters"
            )

        if not self.doi and not any(
            value is not None for value in search_fields
        ):
            raise ValueError(
                "doi, query, title, author, issn, date, or work_type is required"
            )

        if (
            self.from_pub_date is not None
            and self.until_pub_date is not None
            and self.from_pub_date > self.until_pub_date
        ):
            raise ValueError(
                "from_pub_date must be less than or equal to until_pub_date"
            )

        return self
