"""Pydantic schemas shared across ingestion, extraction, matching, and explanation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SeniorityLevel = Literal[
    "intern",
    "analyst",
    "consultant",
    "senior_consultant",
    "manager",
    "principal",
]

SkillCategory = Literal["technical", "functional", "domain", "soft"]

LanguageProficiency = Literal[
    "native",
    "fluent",
    "professional",
    "conversational",
    "basic",
]

TrustClassification = Literal["injection", "promotional_language", "benign"]
TrustSeverity = Literal["low", "medium", "high"]


class TrustFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span: str = Field(min_length=1, description="The matched text that triggered the flag.")
    classification: TrustClassification
    severity: TrustSeverity
    pattern_name: str = Field(description="The stage-1 heuristic that produced this candidate.")


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    category: SkillCategory
    evidence: str = Field(
        min_length=1,
        description="The exact CV sentence supporting this skill. Never empty.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Language(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    proficiency: LanguageProficiency


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    industry: str
    role: str
    impact: str
    tech: list[str] = Field(default_factory=list)
    year_start: int
    year_end: int | None = None  # None for projects still in progress


class ConsultantProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consultant_id: str
    current_role: str
    seniority: SeniorityLevel
    years_experience: int = Field(ge=0)
    skills: list[Skill] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    location: str
    projects: list[Project] = Field(default_factory=list)
    trust_flags: list[str] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class PersonalData(BaseModel):
    """Identifiers stored separately from ConsultantProfile, keyed by consultant_id.

    Never sent to the LLM during matching -- see CLAUDE.md rule 5.
    """

    model_config = ConfigDict(extra="forbid")

    consultant_id: str
    full_name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None


class RoleRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    seniority: SeniorityLevel
    count: int = Field(ge=1)
    required_skills: list[str] = Field(default_factory=list)


class ProjectBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client: str
    industry: str
    roles_needed: list[RoleRequirement] = Field(default_factory=list)
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    start_date: str | None = None  # free text: briefs say "ASAP", "Q3 2026", etc.
    duration_weeks: int | None = Field(default=None, ge=1)
    seniority_mix: dict[SeniorityLevel, int] = Field(default_factory=dict)
    location: str | None = None
    budget: str | None = None  # free text: often a range or "not disclosed"
    inferred_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields the LLM guessed rather than read from the brief.",
    )
