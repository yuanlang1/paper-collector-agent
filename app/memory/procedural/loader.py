from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: tuple[str, ...]
    body: str


def user_skill_directory(root: Path, user_id: str) -> Path:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")
    return root / sha256(normalized_user_id.encode("utf-8")).hexdigest()


def render_skill_document(skill: Skill) -> str:
    triggers = ", ".join(skill.triggers)
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"description: {skill.description}\n"
        f"triggers: {triggers}\n"
        "---\n"
        f"{skill.body.strip()}\n"
    )


class SkillLoader:
    def __init__(
        self,
        directories: list[Path],
        *,
        user_skills_root: Path | None = None,
    ) -> None:
        self.directories = directories
        self.user_skills_root = user_skills_root

    def matching_instructions(
        self,
        user_message: str,
        *,
        user_id: str | None = None,
        max_skills: int = 2,
    ) -> str:
        matched = self._match(
            message=user_message,
            max_skills=max_skills,
            user_id=user_id,
        )
        return "\n\n".join(
            f"### {skill.name}\n{skill.body}"
            for skill in matched
        )

    def _match(
        self,
        *,
        message: str,
        max_skills: int,
        user_id: str | None,
    ) -> list[Skill]:
        normalized_message = message.casefold()
        message_words = set(
            re.findall(r"[a-z0-9]{3,}", normalized_message)
        )

        scored: list[tuple[int, Skill]] = []
        for skill in self._load_skills(user_id=user_id):
            trigger_hits = sum(
                trigger.casefold() in normalized_message
                for trigger in skill.triggers
                if trigger.strip()
            )
            if trigger_hits:
                scored.append((100 + trigger_hits, skill))
                continue

            skill_words = set(
                re.findall(
                    r"[a-z0-9]{3,}",
                    f"{skill.name} {skill.description}".casefold(),
                )
            )
            overlap = len(message_words & skill_words)
            if overlap >= 2:
                scored.append((overlap, skill))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored[:max_skills]]

    def _load_skills(self, *, user_id: str | None) -> list[Skill]:
        skills: list[Skill] = []
        directories = list(self.directories)
        if self.user_skills_root is not None and user_id:
            directories.append(
                user_skill_directory(self.user_skills_root, user_id)
            )
        for directory in directories:
            if not directory.is_dir():
                continue
            for path in directory.rglob("SKILL.md"):
                skill = self._parse(path)
                if skill is not None:
                    skills.append(skill)
        return skills

    @staticmethod
    def _parse(path: Path) -> Skill | None:
        text = path.read_text(encoding="utf-8")
        match = re.match(
            r"^---\n(.*?)\n---\n(.*)$",
            text,
            re.DOTALL,
        )
        if match is None:
            return None

        raw_frontmatter, body = match.groups()
        fields = {
            key.strip(): value.strip().strip("'\"")
            for line in raw_frontmatter.splitlines()
            if ":" in line
            for key, _, value in [line.partition(":")]
        }

        name = fields.get("name", "").strip()
        description = fields.get("description", "").strip()
        if not name or not description:
            return None

        triggers = tuple(
            trigger.strip()
            for trigger in fields.get("triggers", "").split(",")
            if trigger.strip()
        )
        return Skill(
            name=name,
            description=description,
            triggers=triggers,
            body=body.strip(),
        )
