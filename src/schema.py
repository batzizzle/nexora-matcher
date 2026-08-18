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
    required_language: str | None = None  # added for src/match.py's hard-filter stage; see DECISIONS.md phase 3
    budget: str | None = None  # free text: often a range or "not disclosed"
    inferred_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields the LLM guessed rather than read from the brief.",
    )


class RoleFitScore(BaseModel):
    """One LLM-scored verdict for one candidate against one role, with cited evidence (CLAUDE.md rule 4)."""

    model_config = ConfigDict(extra="forbid")

    consultant_id: str
    role_title: str
    fit_score: float = Field(ge=0.0, le=100.0)
    reasons: list[str] = Field(min_length=3, max_length=3)
    concern: str = Field(min_length=1)


class FilterFunnelStage(BaseModel):
    """How many candidates survived one stage of the deterministic hard filter -- for the demo's funnel view."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    survived: int = Field(ge=0)


class TeamMember(BaseModel):
    """One consultant assigned to one role in a proposed team, with the score breakdown that justified picking them."""

    model_config = ConfigDict(extra="forbid")

    role_title: str
    consultant_id: str
    fit_score: float = Field(ge=0.0, le=100.0)
    reasons: list[str] = Field(min_length=3, max_length=3)
    concern: str
    availability_status: str
    free_days_per_week: int = Field(ge=0, le=5)
    next_free_date: str
    assembly_score: float
    adjustments: dict[str, float] = Field(
        default_factory=dict,
        description="Stage-4 score adjustments applied on top of fit_score, e.g. skill_overlap_penalty.",
    )
    selection_note: str = Field(
        default="",
        description="Why this member, specifically, was picked for this team variant (e.g. earliest available).",
    )


class StaffingGap(BaseModel):
    """One role slot a team could not confidently fill -- either left empty, or filled below the fit-confidence bar."""

    model_config = ConfigDict(extra="forbid")

    role_title: str
    reason: Literal["understaffed", "low_confidence_fit"]
    detail: str
    consultant_id: str | None = Field(
        default=None, description="Set when reason is low_confidence_fit -- who was picked despite the weak fit."
    )


class Team(BaseModel):
    """A full proposed staffing team: one member per required role slot, plus what it was optimised for."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["recommended", "earliest_start", "lowest_cost"]
    members: list[TeamMember] = Field(default_factory=list)
    gaps: list[StaffingGap] = Field(
        default_factory=list,
        description="Role slots this team could not confidently fill -- empty means fully, confidently staffed.",
    )


class AvailabilityTradeoff(BaseModel):
    """A candidate who'd otherwise qualify for the project but isn't free until after the requested start date."""

    model_config = ConfigDict(extra="forbid")

    consultant_id: str
    next_free_date: str
    days_after_start: int = Field(
        description="How many days after the brief's requested start date this candidate becomes free."
    )


class MatchResult(BaseModel):
    """Everything src/match.py produces for one project brief: the hard-filter funnel, and three candidate teams."""

    model_config = ConfigDict(extra="forbid")

    funnel: list[FilterFunnelStage]
    availability_tradeoffs: list[AvailabilityTradeoff] = Field(
        default_factory=list,
        description="Candidates dropped only by the availability filter -- the start-date-vs-fit tradeoff, surfaced.",
    )
    teams: list[Team]
    role_rankings: dict[str, list[RoleFitScore]] = Field(
        default_factory=dict,
        description=(
            "Every stage-3-scored candidate per role, not just who was picked -- src/explain.py needs the "
            "full ranking (not just the winner) to build a counterfactual against the next-best alternative."
        ),
    )


TrustBadge = Literal["verified", "flagged", "unverified_claims"]


class ScoreBreakdown(BaseModel):
    """A deterministic, five-component explanation of fit -- complementary to fit_score, not a formula that sums to it.

    fit_score is a single holistic LLM judgement; these five components are separately computed by plain Python
    from structured data (skills, seniority, availability, industries, team composition) so a human can see which
    specific dimension drove -- or hurt -- a recommendation, without re-deriving it from prose.
    """

    model_config = ConfigDict(extra="forbid")

    skill_fit: float = Field(ge=0.0, le=100.0)
    seniority_fit: float = Field(ge=0.0, le=100.0)
    availability: float = Field(ge=0.0, le=100.0)
    industry_experience: float = Field(ge=0.0, le=100.0)
    team_chemistry: float = Field(ge=0.0, le=100.0)


class EvidenceQuote(BaseModel):
    """One verbatim CV sentence backing a skill claim, with a best-effort guess at which project it came from."""

    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1, description="Verbatim, taken directly from Skill.evidence -- never paraphrased.")
    skill_name: str
    project_title: str | None = Field(
        default=None, description="Best-effort attribution via tech-list overlap; None if no project could be matched."
    )
    matched_requirement: bool = Field(
        description=(
            "Whether skill_name is an exact match to one of the role's required_skills or the brief's "
            "must_have_skills. False means this is the candidate's best general skill, shown because rule 4 "
            "requires every score to carry evidence, NOT because it actually relates to what was asked."
        )
    )


class Counterfactual(BaseModel):
    """The next-best candidate for the same role, and a one-sentence, breakdown-delta-generated gain/lose statement."""

    model_config = ConfigDict(extra="forbid")

    consultant_id: str
    fit_score: float = Field(ge=0.0, le=100.0)
    breakdown: ScoreBreakdown
    trust_badge: TrustBadge = Field(
        description="The alternative's own trust badge -- a swap suggestion must never drop this signal."
    )
    summary: str = Field(min_length=1, description="Deterministically templated from score-breakdown deltas, not an LLM call.")


class AvailabilityAlternative(BaseModel):
    """A currently-unavailable candidate whose deterministic breakdown beats the pick for this role -- not LLM-scored.

    Distinct from Counterfactual: this candidate never went through stage 3 (they failed the availability hard
    filter before retrieval/re-ranking ever saw them), so there is no fit_score to report -- only the deterministic
    ScoreBreakdown comparison src/explain.py can compute without them ever having been scored.
    """

    model_config = ConfigDict(extra="forbid")

    consultant_id: str
    next_free_date: str
    days_after_start: int
    breakdown: ScoreBreakdown
    trust_badge: TrustBadge
    summary: str = Field(min_length=1, description="Deterministically templated from score-breakdown deltas, not an LLM call.")


class MatchCard(BaseModel):
    """Everything src/explain.py produces to explain one consultant's fit for one role: score, evidence, trust, alternative."""

    model_config = ConfigDict(extra="forbid")

    consultant_id: str
    role_title: str
    overall_score: float = Field(ge=0.0, le=100.0)
    breakdown: ScoreBreakdown
    evidence: list[EvidenceQuote] = Field(max_length=3)
    trust_badge: TrustBadge
    counterfactual: Counterfactual | None = Field(
        default=None, description="None only when no other ranked candidate exists for this role to compare against."
    )
    availability_alternative: AvailabilityAlternative | None = Field(
        default=None,
        description="A currently-unavailable candidate who'd deterministically beat this pick -- None if nobody does.",
    )
