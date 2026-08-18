"""Turn one selected consultant's team-assembly result into a human-readable match card.

What this does: for one consultant staffed onto one role, produces a match card with
their overall fit score, a five-part breakdown of *why* (skill fit, seniority fit,
availability, industry experience, team chemistry), the top three CV sentences that
back up their skills (quoted verbatim, with a best-effort guess at which project each
came from), a trust badge (verified / flagged / unverified claims), a counterfactual
-- the next-best candidate who wasn't picked for this role, and a one-sentence
statement of what swapping them in would gain and lose -- and an availability
alternative: a currently-unavailable candidate whose deterministic breakdown says
they'd beat the pick, with how many days late they'd be.

Why it exists: src/match.py already decides who gets staffed and scores them, but a
single fit_score number and three LLM-written reasons don't answer "why this specific
number," "who else was close," or "was there someone better who just wasn't free yet."
CLAUDE.md's repo layout scopes this file to "score breakdown, evidence, counterfactual"
-- the presentation/explanation layer that sits between match.py's decision and a
human reading it.

What it takes in / produces: input is one TeamMember plus the Team it's part of, the
ProjectBrief, MatchResult.role_rankings (every stage-3-scored candidate per role, not
just the winner -- needed for the counterfactual), MatchResult.availability_tradeoffs
(candidates who'd otherwise qualify but are booked past the requested start -- needed
for the availability alternative), and the same profiles/availability/co-delivery-graph
data src/match.py already used to build that result. Output is a MatchCard (or a list
of them, one per team member). Nothing here calls the LLM again: every field is either
reused directly from src/match.py's stage-3 output (fit_score, concern) or computed
deterministically from structured data already sitting in ConsultantProfile/
availability.csv/the co-delivery graph -- both the counterfactual and the availability
alternative are explicitly required to come from score-breakdown deltas, not an extra
call. Measured cost of the availability alternative specifically: ~11 microseconds per
compute_breakdown call, so even a brief with hundreds of tradeoff candidates and every
role in a large project adds single-digit milliseconds -- it does not touch the ~60-80s
per-role cost that actually dominates match()'s wall time (src/match.py's docstring).

Assumptions and shortcuts taken:
- The five breakdown components are a deterministic, separately-computed *explanation*
  of fit, not a formula whose weighted sum reproduces fit_score -- fit_score is a
  single holistic LLM judgement; these are plain-Python proxies for five different
  questions a human would ask, computed independently.
- skill_fit and industry_experience reuse the same exact-match (case-insensitive,
  trimmed) convention src/match.py's hard filter already uses for the same reason:
  fuzzy matching risks a false-positive "good fit" signal, which is worse than an
  honest low score on a real match phrased differently.
- team_chemistry is computed fresh against a team's *final* roster (all other members,
  order-independent), not reused from TeamMember.adjustments -- that field only exists
  for the "recommended" team (alt teams leave it empty) and is itself an
  order-dependent snapshot from greedy assembly, not a description of the finished
  team. Recomputing here gives every team variant, and every card, the same
  well-defined "how does this person fit with the rest of THIS team" answer.
- Evidence quotes are taken verbatim from Skill.evidence (schema-guaranteed non-empty,
  extraction-prompted to be a verbatim CV quote -- see src/extract.py) and picked by
  matching the role's/brief's required skills first, then by extraction confidence,
  skipping any skill whose evidence text duplicates one already selected. Verified
  necessary against real data: several CVs quote a whole "Technical Skills:" list
  line as evidence for every skill on that line (a legitimate verbatim quote per
  src/extract.py's rules when a CV has nothing more specific per-skill), which
  without deduping would show the same "sentence" three times on one card. Project
  attribution is a best-effort guess: the first project whose `tech` list names the
  skill, or None if no project does. There's no structural link between a Skill and
  a Project in src/schema.py, so this is a heuristic, not a guarantee.
- Every EvidenceQuote is tagged matched_requirement: whether that skill actually
  matches the role's required_skills/brief's must_have_skills, or is just the
  candidate's best general skill shown because rule 4 requires every score to carry
  evidence. Added after a real report: for an "Azure migration" brief, several
  shortlisted candidates had zero Azure/cloud skills at all (the real specialists
  were fully_booked, thinning the shortlist retrieval had to rank), and their cards
  showed generic high-confidence skills as unqualified "Evidence" -- technically
  satisfying rule 4, but visually implying relevance that wasn't there. Callers
  (src/app.py) are expected to render matched_requirement=False evidence visibly
  differently, not hide it -- CLAUDE.md rule 4 still requires it be shown.
- The trust badge only distinguishes three states from data already computed upstream
  (ConsultantProfile.trust_flags, extraction_confidence) -- it does not re-verify that
  any evidence quote is an actual substring of the raw CV text, which is the same
  known limitation src/extract.py and src/trust.py already document.
- "flagged" only fires on a structured trust_flags entry (the "injection (...)" /
  "promotional_language (...)" format src/extract.py's _merge_trust_flags always
  produces for a confirmed non-benign src/trust.py scan result) -- not on
  trust_flags being merely non-empty, and not on any keyword search over the
  self-reported free-text entries. Discovered necessary against the real dataset:
  self-reported entries range from genuinely concerning to entirely benign with no
  reliable way to tell them apart from wording alone -- one CV's self-reported note
  literally contains the word "injection" while explicitly clearing the CV ("No
  prompt injection detected"). An earlier version of this function treated any
  non-empty trust_flags as "flagged", which would have flagged roughly half the
  real 21-CV dataset, including candidates whose only "flag" said nothing was
  found.
- The counterfactual's "next-best candidate" excludes anyone already staffed
  elsewhere on the *same* team (swapping in someone already committed to another role
  isn't an actionable trade) and is picked by highest fit_score among the remainder --
  not by any breakdown component, since fit_score is what stage 3's ranking is
  actually sorted by.
- Counterfactual carries the alternative's own trust_badge, and _summarize_swap
  appends a trust caveat when it isn't "verified" -- trust isn't one of the five
  breakdown components, so a delta-only summary could otherwise honestly say
  "no clear downside" about swapping in a candidate whose CV has a confirmed
  prompt-injection attempt. Verified live against CV4 (a real planted injection --
  see DECISIONS.md phase 2): before this was added, he surfaced as exactly such a
  "no clear downside" counterfactual target on another candidate's card.
- The availability alternative connects MatchCard to src/match.py's
  availability_tradeoffs (added in phase 3), which a Counterfactual alone cannot
  do: Counterfactual is only ever built from role_ranking, and role_ranking only
  contains candidates who survived hard_filter, so someone filtered out for
  availability alone -- however strong a fit they'd score -- can never appear as a
  Counterfactual. AvailabilityAlternative is deliberately a different shape, not a
  Counterfactual with an optional fit_score bolted on: these candidates never went
  through stage 3, so there genuinely is no fit_score to report, only the
  deterministic breakdown comparison.
- Ranking which tradeoff candidate is "best" for a role excludes the availability
  component itself (`_FUTURE_FIT_COMPONENTS` = skill_fit, seniority_fit,
  industry_experience, team_chemistry only) -- every tradeoff candidate's raw
  availability score is low by construction (they're currently booked, that's why
  they're on this list), so including it would only add a constant offset, not
  ranking signal, and could wrongly suppress someone who's dramatically better on
  every dimension that actually distinguishes candidates.
- An availability alternative is only surfaced when its future-fit rank genuinely
  exceeds the selected member's -- not shown just because a tradeoff candidate
  exists. A card showing "here's someone worse who's also unavailable" on every
  role would be noise, not signal.
- Both build_counterfactual and build_availability_alternative share the same
  underlying delta-to-sentence logic (_delta_summary_sentence) and only differ in
  which caveat gets appended (a trust caveat for both when not "verified"; an
  additional "not free for N days" caveat for the availability alternative) --
  kept as one shared function rather than two near-duplicate implementations.
"""

from __future__ import annotations

from src.schema import (
    AvailabilityAlternative,
    AvailabilityTradeoff,
    ConsultantProfile,
    Counterfactual,
    EvidenceQuote,
    MatchCard,
    Project,
    ProjectBrief,
    RoleFitScore,
    RoleRequirement,
    ScoreBreakdown,
    Team,
    TeamMember,
    TrustBadge,
)

_SENIORITY_RANK = {  # duplicated from src/availability.py / src/match.py -- see those files' docstrings on why
    "intern": 0,
    "analyst": 1,
    "consultant": 2,
    "senior_consultant": 3,
    "manager": 4,
    "principal": 5,
}

_EVIDENCE_COUNT = 3  # fixed by the phase-4 brief: "top three evidence sentences"

# --- score breakdown: tunable PoC weights, none validated against a labelled dataset ---
_SENIORITY_MISMATCH_PENALTY_PER_TIER = 20.0
_AVAILABLE_SCORE = 100.0
_PARTLY_BOOKED_BASE_SCORE = 50.0  # score at (hypothetical) 0 free days/week
_PARTLY_BOOKED_SCORE_PER_FREE_DAY = 10.0
_FULLY_BOOKED_SCORE = 60.0  # they passed the hard filter (free by the start date) but have zero current slack
_INDUSTRY_MATCH_SCORE = 80.0
_INDUSTRY_NO_MATCH_SCORE = 25.0
_CHEMISTRY_BASELINE_SCORE = 70.0  # neutral starting point before co-delivery bonus / skill-overlap penalty
_CHEMISTRY_BONUS_PER_SHARED_PROJECT = 10.0
_CHEMISTRY_PENALTY_PER_OVERLAPPING_SKILL = 6.0

_UNVERIFIED_CONFIDENCE_THRESHOLD = 0.7  # tunable PoC cutoff, not validated

_COMPONENT_LABELS = {
    "skill_fit": "skill fit",
    "seniority_fit": "seniority fit",
    "availability": "availability",
    "industry_experience": "industry experience",
    "team_chemistry": "team chemistry",
}


def _skill_names(profile: ConsultantProfile) -> set[str]:
    """A candidate's skill names, normalised for set comparison -- duplicated from src/match.py, same PoC rationale."""
    return {s.name.strip().lower() for s in profile.skills}


def _skill_fit(profile: ConsultantProfile, role: RoleRequirement, brief: ProjectBrief) -> float:
    """What fraction of the role's + brief's required skills this candidate has, by exact case-insensitive name match."""
    required = {s.strip().lower() for s in [*role.required_skills, *brief.must_have_skills]}
    if not required:
        return 100.0  # nothing specifically required -- trivially satisfied, not evidence of unusual strength
    matched = required & _skill_names(profile)
    return round(100.0 * len(matched) / len(required), 1)


def _seniority_fit(profile: ConsultantProfile, role: RoleRequirement) -> float:
    """How close the candidate's seniority tier is to the role's requested tier, penalised per tier of distance."""
    delta = abs(_SENIORITY_RANK.get(profile.seniority, 0) - _SENIORITY_RANK.get(role.seniority, 0))
    return max(0.0, 100.0 - _SENIORITY_MISMATCH_PENALTY_PER_TIER * delta)


def _availability_score(status: str, free_days_per_week: int) -> float:
    """Score a candidate's current bench status -- available highest, partly_booked scaled by free days, else flat."""
    if status == "available":
        return _AVAILABLE_SCORE
    if status == "partly_booked":
        return min(100.0, _PARTLY_BOOKED_BASE_SCORE + _PARTLY_BOOKED_SCORE_PER_FREE_DAY * free_days_per_week)
    return _FULLY_BOOKED_SCORE  # fully_booked (or an unrecognised status) but survived the hard filter regardless


def _industry_experience(profile: ConsultantProfile, brief: ProjectBrief) -> float:
    """Whether the candidate lists an industry matching the brief's, by substring either direction."""
    if not brief.industry:
        return 100.0
    target = brief.industry.strip().lower()
    matches = any(target in i.strip().lower() or i.strip().lower() in target for i in profile.industries)
    return _INDUSTRY_MATCH_SCORE if matches else _INDUSTRY_NO_MATCH_SCORE


def _team_chemistry(profile: ConsultantProfile, teammates: list[ConsultantProfile], graph: dict[str, dict[str, int]]) -> float:
    """How well this candidate complements a specific set of teammates: co-delivery history rewarded, skill overlap penalised."""
    teammate_skills: set[str] = set()
    for teammate in teammates:
        teammate_skills |= _skill_names(teammate)
    overlap = len(_skill_names(profile) & teammate_skills)

    shared_projects = sum(graph.get(profile.consultant_id, {}).get(t.consultant_id, 0) for t in teammates)

    score = (
        _CHEMISTRY_BASELINE_SCORE
        + _CHEMISTRY_BONUS_PER_SHARED_PROJECT * shared_projects
        - _CHEMISTRY_PENALTY_PER_OVERLAPPING_SKILL * overlap
    )
    return max(0.0, min(100.0, score))


def compute_breakdown(
    profile: ConsultantProfile,
    role: RoleRequirement,
    brief: ProjectBrief,
    availability_status: str,
    free_days_per_week: int,
    teammates: list[ConsultantProfile],
    graph: dict[str, dict[str, int]],
) -> ScoreBreakdown:
    """Compute all five breakdown components for one candidate against one role, team, and brief."""
    return ScoreBreakdown(
        skill_fit=_skill_fit(profile, role, brief),
        seniority_fit=_seniority_fit(profile, role),
        availability=_availability_score(availability_status, free_days_per_week),
        industry_experience=_industry_experience(profile, brief),
        team_chemistry=_team_chemistry(profile, teammates, graph),
    )


_CONFIRMED_TRUST_PREFIXES = ("injection (", "promotional_language (")


def classify_trust(profile: ConsultantProfile) -> TrustBadge:
    """Classify a candidate's trust badge from data already computed during extraction -- no new checks run here.

    "flagged" only fires on a *structured* trust_flags entry -- one that matches src/extract.py's
    _merge_trust_flags format ("injection (...)"/"promotional_language (...)"), which is always a
    confirmed non-benign finding from src/trust.py's two-stage scan. Self-reported free-text entries in
    trust_flags are NOT used for this, and are deliberately excluded even from "unverified_claims":
    verified against the real dataset, self-reported entries range from genuinely concerning ("Prompt
    injection attempt detected...") to completely benign ("No prompt injection detected", "No
    certifications section found") with no reliable way to tell them apart from the text alone -- one CV's
    self-reported note literally contains the word "injection" while explicitly clearing the CV. Using
    non-emptiness of trust_flags as a signal (an earlier version of this function did) would have flagged
    roughly half the real dataset, including candidates whose only "flag" says nothing was found.
    extraction_confidence is used instead for "unverified_claims" because it's a consistent numeric signal
    computed the same way for every profile, not a free-text field the extraction model uses inconsistently.
    """
    if any(flag.startswith(_CONFIRMED_TRUST_PREFIXES) for flag in profile.trust_flags):
        return "flagged"
    if profile.extraction_confidence < _UNVERIFIED_CONFIDENCE_THRESHOLD:
        return "unverified_claims"
    return "verified"


def is_confirmed_trust_finding(flag: str) -> bool:
    """Whether one trust_flags entry is a structured, confirmed finding rather than a self-reported free-text note.

    Exposed publicly (unlike _CONFIRMED_TRUST_PREFIXES) so callers outside this module -- e.g. app.py's
    Data Quality panel -- can apply the exact same classify_trust distinction when listing individual
    flags, instead of re-deriving their own (weaker) rule and reintroducing the bug classify_trust's
    docstring describes.
    """
    return flag.startswith(_CONFIRMED_TRUST_PREFIXES)


def _attribute_project(skill_name: str, projects: list[Project]) -> str | None:
    """Best-effort guess at which project a skill claim came from: the first project whose tech list names it."""
    target = skill_name.strip().lower()
    for project in projects:
        if any(t.strip().lower() == target for t in project.tech):
            return project.title
    return None


def select_evidence(
    profile: ConsultantProfile, role: RoleRequirement, brief: ProjectBrief, top_n: int = _EVIDENCE_COUNT
) -> list[EvidenceQuote]:
    """Pick the top verbatim skill-evidence quotes: required skills first, then by extraction confidence.

    Skips skills whose evidence text was already used by a higher-ranked skill -- some CVs quote a whole
    "Technical Skills:" list line as evidence for every skill on it (verified against real extracted data:
    src/extract.py's prompt requires a verbatim quote, and a skills-list line is a legitimate verbatim quote
    when the CV has nothing more specific per-skill). Without deduping, a card could show the same "sentence"
    three times, which defeats the point of three *distinct* pieces of evidence.

    Every returned quote is tagged matched_requirement (see EvidenceQuote) so a candidate with zero skills
    matching what was actually asked for still gets evidence -- CLAUDE.md rule 4 requires every score to
    carry evidence, so this never returns fewer than min(top_n, len(profile.skills)) -- but nothing is
    silently presented as relevant when it isn't. Discovered necessary from a real report: for an "Azure
    migration" brief, several candidates with zero Azure/cloud skills were shortlisted (retrieval imprecision
    against a shortlist thinned by availability -- the real Azure specialists were fully_booked), and their
    cards showed generic high-confidence skills as "Evidence" with no indication they had nothing to do with
    Azure. The card was technically accurate (rule 4 was satisfied) but visually misleading.
    """
    required = {s.strip().lower() for s in [*role.required_skills, *brief.must_have_skills]}
    ranked = sorted(
        profile.skills,
        key=lambda skill: (skill.name.strip().lower() in required, skill.confidence),
        reverse=True,
    )

    quotes: list[EvidenceQuote] = []
    seen_evidence: set[str] = set()
    for skill in ranked:
        if len(quotes) >= top_n:
            break
        if skill.evidence in seen_evidence:
            continue
        seen_evidence.add(skill.evidence)
        quotes.append(
            EvidenceQuote(
                quote=skill.evidence,
                skill_name=skill.name,
                project_title=_attribute_project(skill.name, profile.projects),
                matched_requirement=skill.name.strip().lower() in required,
            )
        )
    return quotes


def _breakdown_deltas(alternative: ScoreBreakdown, selected: ScoreBreakdown) -> dict[str, float]:
    """Per-component (alternative - selected) deltas -- positive means the alternative is better on that component."""
    return {field: round(getattr(alternative, field) - getattr(selected, field), 1) for field in ScoreBreakdown.model_fields}


_TRUST_CAVEATS: dict[TrustBadge, str] = {
    "flagged": " Note: this candidate's CV has a flagged trust issue -- see their trust badge before acting on this.",
    "unverified_claims": " Note: this candidate's extraction confidence is low -- their claims are less verified.",
}


def _delta_summary_sentence(alternative_id: str, deltas: dict[str, float]) -> str:
    """Deterministically template a one-sentence gain/lose statement from breakdown deltas -- no LLM call, no caveats.

    Shared by build_counterfactual and build_availability_alternative, which each append a different caveat
    on top of this core sentence (see _summarize_swap / _summarize_availability_alternative).
    """
    gains = {k: v for k, v in deltas.items() if v > 0}
    losses = {k: v for k, v in deltas.items() if v < 0}

    if not losses:
        label, delta = max(deltas.items(), key=lambda kv: kv[1])
        return f"Swapping in {alternative_id} would improve every component, most notably {_COMPONENT_LABELS[label]} (+{delta:.0f}), with no clear downside."
    if not gains:
        label, delta = min(deltas.items(), key=lambda kv: kv[1])
        return f"Swapping in {alternative_id} would be a strict downgrade, weakest on {_COMPONENT_LABELS[label]} ({delta:.0f})."

    gain_label, gain_delta = max(gains.items(), key=lambda kv: kv[1])
    loss_label, loss_delta = min(losses.items(), key=lambda kv: kv[1])
    return (
        f"Swapping in {alternative_id} would improve {_COMPONENT_LABELS[gain_label]} (+{gain_delta:.0f}) "
        f"but reduce {_COMPONENT_LABELS[loss_label]} ({loss_delta:.0f})."
    )


def _summarize_swap(alternative_id: str, deltas: dict[str, float], trust_badge: TrustBadge) -> str:
    """One-sentence gain/lose statement for an in-pool counterfactual, plus a trust caveat when not "verified".

    Trust isn't one of the five breakdown components, so a delta-only summary could otherwise call a flagged
    candidate's swap "no clear downside", which is exactly the gap that motivated adding trust_badge to
    Counterfactual in the first place.
    """
    return _delta_summary_sentence(alternative_id, deltas) + _TRUST_CAVEATS.get(trust_badge, "")


def _summarize_availability_alternative(
    alternative_id: str, deltas: dict[str, float], days_after_start: int, trust_badge: TrustBadge
) -> str:
    """One-sentence gain/lose statement for a currently-unavailable alternative, plus a not-free-yet caveat."""
    day_word = "day" if abs(days_after_start) == 1 else "days"
    availability_note = f" Not free until {days_after_start} {day_word} after your requested start."
    return _delta_summary_sentence(alternative_id, deltas) + availability_note + _TRUST_CAVEATS.get(trust_badge, "")


def build_counterfactual(
    role_ranking: list[RoleFitScore],
    selected_consultant_id: str,
    excluded_ids: set[str],
    role: RoleRequirement,
    brief: ProjectBrief,
    profiles_by_id: dict[str, ConsultantProfile],
    availability_by_id: dict,
    teammates: list[ConsultantProfile],
    graph: dict[str, dict[str, int]],
    selected_breakdown: ScoreBreakdown,
) -> Counterfactual | None:
    """Build the next-best-candidate counterfactual for one role, or None if nobody else was ranked and available."""
    alternatives = [
        entry for entry in role_ranking if entry.consultant_id != selected_consultant_id and entry.consultant_id not in excluded_ids
    ]
    if not alternatives:
        return None

    best = max(alternatives, key=lambda entry: entry.fit_score)
    alt_profile = profiles_by_id[best.consultant_id]
    alt_row = availability_by_id[best.consultant_id]
    alt_breakdown = compute_breakdown(
        alt_profile, role, brief, alt_row["status"], int(alt_row["free_days_per_week"]), teammates, graph
    )
    deltas = _breakdown_deltas(alt_breakdown, selected_breakdown)
    alt_trust_badge = classify_trust(alt_profile)

    return Counterfactual(
        consultant_id=best.consultant_id,
        fit_score=best.fit_score,
        breakdown=alt_breakdown,
        trust_badge=alt_trust_badge,
        summary=_summarize_swap(best.consultant_id, deltas, alt_trust_badge),
    )


# Deliberately excludes "availability" -- every tradeoff candidate's availability score is low by
# construction (they're currently booked, that's why they're on this list), so including it would
# only add a constant offset, not ranking signal.
_FUTURE_FIT_COMPONENTS = ("skill_fit", "seniority_fit", "industry_experience", "team_chemistry")


def _future_fit_rank(breakdown: ScoreBreakdown) -> float:
    """Deterministic ranking key for comparing currently-unavailable candidates: mean of the non-availability components."""
    return sum(getattr(breakdown, component) for component in _FUTURE_FIT_COMPONENTS) / len(_FUTURE_FIT_COMPONENTS)


def build_availability_alternative(
    tradeoffs: list[AvailabilityTradeoff],
    role: RoleRequirement,
    brief: ProjectBrief,
    profiles_by_id: dict[str, ConsultantProfile],
    availability_by_id: dict,
    teammates: list[ConsultantProfile],
    graph: dict[str, dict[str, int]],
    selected_breakdown: ScoreBreakdown,
) -> AvailabilityAlternative | None:
    """Find the strongest currently-unavailable candidate for this role and compare them to the pick.

    Every candidate here already failed src/match.py's availability hard filter, so they were never a
    Counterfactual; tradeoffs already excludes anyone who survived the filter, so no excluded_ids set is
    needed (unlike build_counterfactual). Returns None unless the best one genuinely outranks the pick --
    a card showing "here's someone worse who's also unavailable" would be noise, not signal.
    """
    if not tradeoffs:
        return None

    scored = []
    for tradeoff in tradeoffs:
        profile = profiles_by_id[tradeoff.consultant_id]
        row = availability_by_id[tradeoff.consultant_id]
        breakdown = compute_breakdown(
            profile, role, brief, row["status"], int(row["free_days_per_week"]), teammates, graph
        )
        scored.append((tradeoff, breakdown))

    best_tradeoff, best_breakdown = max(scored, key=lambda pair: _future_fit_rank(pair[1]))
    if _future_fit_rank(best_breakdown) <= _future_fit_rank(selected_breakdown):
        return None

    deltas = _breakdown_deltas(best_breakdown, selected_breakdown)
    trust_badge = classify_trust(profiles_by_id[best_tradeoff.consultant_id])

    return AvailabilityAlternative(
        consultant_id=best_tradeoff.consultant_id,
        next_free_date=best_tradeoff.next_free_date,
        days_after_start=best_tradeoff.days_after_start,
        breakdown=best_breakdown,
        trust_badge=trust_badge,
        summary=_summarize_availability_alternative(
            best_tradeoff.consultant_id, deltas, best_tradeoff.days_after_start, trust_badge
        ),
    )


def build_match_card(
    member: TeamMember,
    team: Team,
    brief: ProjectBrief,
    role_rankings: dict[str, list[RoleFitScore]],
    profiles_by_id: dict[str, ConsultantProfile],
    availability_by_id: dict,
    graph: dict[str, dict[str, int]],
    availability_tradeoffs: list[AvailabilityTradeoff] = (),
) -> MatchCard:
    """Build one consultant's match card: score, breakdown, evidence, trust badge, counterfactual, availability alternative."""
    roles_by_title = {r.title: r for r in brief.roles_needed}
    role = roles_by_title[member.role_title]
    profile = profiles_by_id[member.consultant_id]
    teammates = [profiles_by_id[m.consultant_id] for m in team.members if m.consultant_id != member.consultant_id]

    breakdown = compute_breakdown(
        profile, role, brief, member.availability_status, member.free_days_per_week, teammates, graph
    )
    excluded_ids = {m.consultant_id for m in team.members}
    counterfactual = build_counterfactual(
        role_rankings.get(member.role_title, []),
        member.consultant_id,
        excluded_ids,
        role,
        brief,
        profiles_by_id,
        availability_by_id,
        teammates,
        graph,
        breakdown,
    )
    availability_alternative = build_availability_alternative(
        list(availability_tradeoffs), role, brief, profiles_by_id, availability_by_id, teammates, graph, breakdown
    )

    return MatchCard(
        consultant_id=member.consultant_id,
        role_title=member.role_title,
        overall_score=member.fit_score,
        breakdown=breakdown,
        evidence=select_evidence(profile, role, brief),
        trust_badge=classify_trust(profile),
        counterfactual=counterfactual,
        availability_alternative=availability_alternative,
    )


def build_match_cards_for_team(
    team: Team,
    brief: ProjectBrief,
    role_rankings: dict[str, list[RoleFitScore]],
    profiles_by_id: dict[str, ConsultantProfile],
    availability_by_id: dict,
    graph: dict[str, dict[str, int]],
    availability_tradeoffs: list[AvailabilityTradeoff] = (),
) -> list[MatchCard]:
    """Build a match card for every member of one team."""
    return [
        build_match_card(
            member, team, brief, role_rankings, profiles_by_id, availability_by_id, graph, availability_tradeoffs
        )
        for member in team.members
    ]
