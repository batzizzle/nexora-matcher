"""Define the shapes of data every other module reads and writes.

What this does: declares, in one place, exactly what a "consultant profile,"
a "project brief," a "trust flag," and so on look like -- what fields they
have, what type each field must be, and what values are and aren't allowed.

Why it exists: several parts of this project ask an AI model to read a CV or
a project brief and turn it into structured data. AI output can't be trusted
blindly -- it might invent a field, use the wrong type, or return a value
outside the allowed range. Every model call in this project is required to
produce output that fits one of the shapes defined here; if it doesn't, that
call fails loudly instead of quietly poisoning the rest of the pipeline with
bad data.

What it takes in / produces: this file defines shapes only -- it doesn't run
any logic itself. Other modules import these definitions to validate the
data they build or receive.

Assumptions and shortcuts taken:
- Every shape rejects unrecognised fields outright (`extra="forbid"`),
  because a stray field from an AI response is more likely a hallucination
  than a useful extra.
- Some fields that could plausibly be missing from a CV or a project brief
  (e.g. a consultant's location, a brief's budget) are still required or
  left un-flagged as inferred, because deciding exactly which fields are
  safe to guess versus must be read verbatim is a judgment call left to the
  extraction prompts (src/extract.py, src/brief.py), not to this schema.
"""

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
    """One suspicious piece of CV text, with what it looked like and how serious it is."""

    model_config = ConfigDict(extra="forbid")

    span: str = Field(min_length=1, description="The matched text that triggered the flag.")
    classification: TrustClassification
    severity: TrustSeverity
    pattern_name: str = Field(description="The stage-1 heuristic that produced this candidate.")


class Skill(BaseModel):
    """One skill a consultant has, backed by the CV sentence that proves it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: SkillCategory
    evidence: str = Field(
        min_length=1,
        description="The exact CV sentence supporting this skill. Never empty.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Language(BaseModel):
    """One language a consultant speaks, and how well."""

    model_config = ConfigDict(extra="forbid")

    name: str
    proficiency: LanguageProficiency


class Project(BaseModel):
    """One past project a consultant worked on, as listed on their CV."""

    model_config = ConfigDict(extra="forbid")

    title: str
    industry: str
    role: str
    impact: str
    tech: list[str] = Field(default_factory=list)
    year_start: int
    year_end: int | None = None  # None for projects still in progress


class ConsultantProfile(BaseModel):
    """Everything the matcher knows about one consultant, with no personal identifiers included."""

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
    """One role a client's project needs staffed, e.g. "2 senior consultants"."""

    model_config = ConfigDict(extra="forbid")

    title: str
    seniority: SeniorityLevel
    count: int = Field(ge=1)
    required_skills: list[str] = Field(default_factory=list)


class ProjectBrief(BaseModel):
    """A client's staffing request, turned from free-text into structured fields the matcher can filter on."""

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
