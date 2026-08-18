import threading
from datetime import date

from src.match import (
    _min_max_normalize,
    _parse_start_date,
    _passes_availability,
    _passes_language,
    _passes_location,
    _passes_must_have_skills,
    _role_fill_order,
    _staffing_gaps,
    assemble_earliest_start_team,
    assemble_lowest_cost_team,
    assemble_team,
    availability_tradeoffs,
    hard_filter,
    hybrid_retrieve,
    match,
    rerank_role,
)
from src.schema import ConsultantProfile, ProjectBrief, RoleFitScore, RoleRequirement, TeamMember

TODAY = date(2026, 3, 1)

BASE_PROFILE = dict(
    current_role="Consultant",
    seniority="consultant",
    years_experience=5,
    location="Copenhagen, Denmark",
    extraction_confidence=0.9,
)

REASONS = ["reason one", "reason two", "reason three"]


def make_profile(consultant_id, **overrides):
    return ConsultantProfile(**{**BASE_PROFILE, "consultant_id": consultant_id, **overrides})


def make_skill(name):
    return {"name": name, "category": "technical", "evidence": f"Used {name} on a project.", "confidence": 0.9}


def make_language(name, proficiency="fluent"):
    return {"name": name, "proficiency": proficiency}


def make_project(title, tech=None, role="Consultant"):
    return {
        "title": title,
        "industry": "Financial Services",
        "role": role,
        "impact": "Delivered on time and under budget.",
        "tech": tech or [],
        "year_start": 2023,
    }


def make_brief(**overrides):
    defaults = dict(client="Acme", industry="Retail", roles_needed=[])
    return ProjectBrief(**{**defaults, **overrides})


def make_fit_score(consultant_id, role_title, fit_score, reasons=None, concern="A minor gap."):
    return RoleFitScore(
        consultant_id=consultant_id,
        role_title=role_title,
        fit_score=fit_score,
        reasons=reasons or REASONS,
        concern=concern,
    )


def availability_row(consultant_id, status="available", free_days_per_week=5, next_free_date=None, current_project=""):
    return {
        "consultant_id": consultant_id,
        "status": status,
        "free_days_per_week": free_days_per_week,
        "next_free_date": (next_free_date or TODAY.isoformat()),
        "current_project": current_project,
    }


class _FakeToolUseBlock:
    def __init__(self, input_dict):
        self.type = "tool_use"
        self.input = input_dict


class _FakeResponse:
    def __init__(self, input_dict):
        self.content = [_FakeToolUseBlock(input_dict)]


class _FakeMessages:
    def __init__(self, inputs):
        self._inputs = list(inputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._inputs.pop(0))


class _FakeClient:
    def __init__(self, inputs):
        self.messages = _FakeMessages(inputs)


class _RoleAwareFakeMessages:
    """Picks its canned response by role title in the prompt, not call order -- needed once calls run concurrently."""

    def __init__(self, responses_by_role_title):
        self._responses = responses_by_role_title
        self.calls = []
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        user_content = kwargs["messages"][0]["content"]
        for role_title, response in self._responses.items():
            if f"Role to fill: {role_title}" in user_content:
                return _FakeResponse(response)
        raise AssertionError(f"no fake response configured for message: {user_content[:200]}")


class _RoleAwareFakeClient:
    def __init__(self, responses_by_role_title):
        self.messages = _RoleAwareFakeMessages(responses_by_role_title)


# --- _parse_start_date -------------------------------------------------


def test_parse_start_date_none_and_asap_synonyms_mean_today():
    assert _parse_start_date(None, TODAY) == TODAY
    assert _parse_start_date("ASAP", TODAY) == TODAY
    assert _parse_start_date("immediately", TODAY) == TODAY


def test_parse_start_date_handles_iso_quarter_and_month_year():
    assert _parse_start_date("2026-07-15", TODAY) == date(2026, 7, 15)
    assert _parse_start_date("Q3 2026", TODAY) == date(2026, 7, 1)
    assert _parse_start_date("March 2026", TODAY) == date(2026, 3, 1)


def test_parse_start_date_returns_none_for_unparseable_text():
    assert _parse_start_date("sometime next year", TODAY) is None


# --- Stage 1: hard filter -----------------------------------------------


def test_passes_availability_false_when_no_availability_row():
    profile = make_profile("p1")
    brief = make_brief()
    assert _passes_availability(profile, brief, {}, TODAY) is False


def test_passes_availability_true_when_free_before_start_date():
    profile = make_profile("p1")
    brief = make_brief(start_date="2026-06-01")
    availability_by_id = {"p1": availability_row("p1", status="fully_booked", next_free_date="2026-05-01")}
    assert _passes_availability(profile, brief, availability_by_id, TODAY) is True


def test_passes_availability_false_when_free_after_start_date():
    profile = make_profile("p1")
    brief = make_brief(start_date="2026-04-01")
    availability_by_id = {"p1": availability_row("p1", status="fully_booked", next_free_date="2026-06-01")}
    assert _passes_availability(profile, brief, availability_by_id, TODAY) is False


def test_passes_language_requires_exact_case_insensitive_match():
    profile = make_profile("p1", languages=[make_language("Danish")])
    assert _passes_language(profile, make_brief(required_language="danish")) is True
    assert _passes_language(profile, make_brief(required_language="German")) is False
    assert _passes_language(profile, make_brief(required_language=None)) is True


def test_passes_must_have_skills_requires_all_skills_present():
    profile = make_profile("p1", skills=[make_skill("Python"), make_skill("SQL")])
    assert _passes_must_have_skills(profile, make_brief(must_have_skills=["Python"])) is True
    assert _passes_must_have_skills(profile, make_brief(must_have_skills=["Python", "SQL"])) is True
    assert _passes_must_have_skills(profile, make_brief(must_have_skills=["Python", "Spark"])) is False


def test_passes_location_matches_substring_either_direction():
    profile = make_profile("p1", location="Aarhus, Denmark")
    assert _passes_location(profile, make_brief(location="Denmark")) is True
    assert _passes_location(profile, make_brief(location="Aarhus, Denmark, Europe")) is True
    assert _passes_location(profile, make_brief(location="Sweden")) is False
    assert _passes_location(profile, make_brief(location=None)) is True


def test_hard_filter_funnel_reports_survivors_at_every_stage_in_order():
    survivor = make_profile("survivor", skills=[make_skill("Python")], languages=[make_language("English")])
    wrong_skill = make_profile("wrong_skill", skills=[make_skill("Java")], languages=[make_language("English")])
    unavailable = make_profile("unavailable", skills=[make_skill("Python")], languages=[make_language("English")])

    brief = make_brief(must_have_skills=["Python"], required_language="English")
    availability_by_id = {
        "survivor": availability_row("survivor"),
        "wrong_skill": availability_row("wrong_skill"),
        "unavailable": availability_row("unavailable", status="fully_booked", next_free_date="2027-01-01"),
    }

    pool, funnel = hard_filter([survivor, wrong_skill, unavailable], brief, availability_by_id, TODAY)

    assert [p.consultant_id for p in pool] == ["survivor"]
    stages = {stage.stage: stage.survived for stage in funnel}
    assert stages["total"] == 3
    assert stages["availability"] == 2  # unavailable dropped
    assert stages["required_language"] == 2
    assert stages["must_have_skills"] == 1  # wrong_skill dropped
    assert stages["location"] == 1


def test_availability_tradeoffs_lists_a_qualified_but_unavailable_candidate():
    profile = make_profile("booked", skills=[make_skill("Python")], languages=[make_language("English")])
    brief = make_brief(must_have_skills=["Python"], required_language="English", start_date="2026-03-01")
    availability_by_id = {"booked": availability_row("booked", status="fully_booked", next_free_date="2026-03-15")}

    tradeoffs = availability_tradeoffs([profile], brief, availability_by_id, TODAY)

    assert len(tradeoffs) == 1
    assert tradeoffs[0].consultant_id == "booked"
    assert tradeoffs[0].next_free_date == "2026-03-15"
    assert tradeoffs[0].days_after_start == 14


def test_availability_tradeoffs_excludes_candidates_already_eligible():
    profile = make_profile("free", skills=[make_skill("Python")], languages=[make_language("English")])
    brief = make_brief(must_have_skills=["Python"], required_language="English")
    availability_by_id = {"free": availability_row("free")}

    assert availability_tradeoffs([profile], brief, availability_by_id, TODAY) == []


def test_availability_tradeoffs_excludes_candidates_who_also_fail_another_filter():
    profile = make_profile("booked_and_wrong_skill", skills=[make_skill("Java")], languages=[make_language("English")])
    brief = make_brief(must_have_skills=["Python"], required_language="English")
    availability_by_id = {
        "booked_and_wrong_skill": availability_row("booked_and_wrong_skill", status="fully_booked", next_free_date="2026-06-01")
    }

    assert availability_tradeoffs([profile], brief, availability_by_id, TODAY) == []


def test_availability_tradeoffs_excludes_candidates_with_no_availability_data():
    profile = make_profile("unknown_availability")
    assert availability_tradeoffs([profile], make_brief(), {}, TODAY) == []


def test_availability_tradeoffs_returns_empty_when_start_date_unparseable():
    profile = make_profile("booked")
    brief = make_brief(start_date="sometime next year")
    availability_by_id = {"booked": availability_row("booked", status="fully_booked", next_free_date="2026-06-01")}

    assert availability_tradeoffs([profile], brief, availability_by_id, TODAY) == []


def test_availability_tradeoffs_sorted_by_next_free_date():
    later = make_profile("later")
    sooner = make_profile("sooner")
    availability_by_id = {
        "later": availability_row("later", status="fully_booked", next_free_date="2026-08-01"),
        "sooner": availability_row("sooner", status="fully_booked", next_free_date="2026-04-01"),
    }

    tradeoffs = availability_tradeoffs([later, sooner], make_brief(start_date="2026-03-01"), availability_by_id, TODAY)

    assert [t.consultant_id for t in tradeoffs] == ["sooner", "later"]


# --- Stage 2: hybrid retrieval -------------------------------------------


def _fixed_embed_fn(vectors_by_text):
    def embed_fn(texts):
        return [vectors_by_text[text] for text in texts]

    return embed_fn


def test_hybrid_retrieve_ranks_strong_match_above_weak_match():
    role = RoleRequirement(title="Data Engineer", seniority="senior_consultant", count=1, required_skills=["Python", "Spark"])
    brief = make_brief(must_have_skills=["Python"])

    strong = make_profile(
        "strong",
        skills=[make_skill("Python"), make_skill("Spark")],
        industries=["Retail"],
        projects=[make_project("Data Platform Build", tech=["Python", "Spark"], role="Data Engineer")],
    )
    weak = make_profile("weak", skills=[make_skill("Excel")], industries=["Legal"])

    query_text = (
        "Data Engineer, seniority: senior_consultant, required skills: Python, Spark. "
        "Client industry: Retail. Project must-have skills: Python."
    )
    embed_fn = _fixed_embed_fn(
        {
            query_text: [1.0, 0.0],
            _profile_summary(strong): [1.0, 0.0],
            _profile_summary(weak): [-1.0, 0.0],
        }
    )

    ranked = hybrid_retrieve([weak, strong], role, brief, embed_fn=embed_fn)

    assert [p.consultant_id for p in ranked] == ["strong", "weak"]


def test_hybrid_retrieve_respects_top_k():
    role = RoleRequirement(title="Analyst", seniority="consultant", count=1, required_skills=[])
    brief = make_brief()
    candidates = [make_profile(f"c{i}") for i in range(5)]

    embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)  # noqa: E731 -- tied vectors isolate top_k behaviour

    ranked = hybrid_retrieve(candidates, role, brief, top_k=2, embed_fn=embed_fn)

    assert len(ranked) == 2


def test_hybrid_retrieve_returns_empty_for_no_candidates():
    role = RoleRequirement(title="Analyst", seniority="consultant", count=1, required_skills=[])
    assert hybrid_retrieve([], role, make_brief()) == []


def _profile_summary(profile):
    from src.match import _profile_summary_text

    return _profile_summary_text(profile)


# --- pure helpers ----------------------------------------------------------


def test_min_max_normalize_scales_to_unit_range():
    assert _min_max_normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_handles_tied_values():
    assert _min_max_normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


# --- Stage 3: LLM re-rank ---------------------------------------------------


def test_rerank_role_maps_llm_response_sorted_by_fit_score():
    role = RoleRequirement(title="Dev", seniority="consultant", count=1, required_skills=[])
    brief = make_brief()
    candidates = [make_profile("cv1"), make_profile("cv2")]
    client = _FakeClient(
        [
            {
                "rankings": [
                    {"consultant_id": "cv1", "fit_score": 60, "reasons": REASONS, "concern": "Limited depth."},
                    {"consultant_id": "cv2", "fit_score": 90, "reasons": REASONS, "concern": "None major."},
                ]
            }
        ]
    )

    scores = rerank_role(role, candidates, brief, client=client)

    assert [s.consultant_id for s in scores] == ["cv2", "cv1"]
    assert scores[0].role_title == "Dev"
    assert len(client.messages.calls) == 1


def test_rerank_role_drops_hallucinated_consultant_id():
    role = RoleRequirement(title="Dev", seniority="consultant", count=1, required_skills=[])
    brief = make_brief()
    candidates = [make_profile("cv1")]
    client = _FakeClient(
        [
            {
                "rankings": [
                    {"consultant_id": "cv1", "fit_score": 80, "reasons": REASONS, "concern": "None."},
                    {"consultant_id": "ghost", "fit_score": 99, "reasons": REASONS, "concern": "None."},
                ]
            }
        ]
    )

    scores = rerank_role(role, candidates, brief, client=client)

    assert [s.consultant_id for s in scores] == ["cv1"]


def test_rerank_role_returns_empty_for_no_candidates():
    role = RoleRequirement(title="Dev", seniority="consultant", count=1, required_skills=[])
    assert rerank_role(role, [], make_brief()) == []


# --- Stage 4: team assembly -------------------------------------------------


def test_role_fill_order_prioritises_seniority_then_scarcity():
    lead = RoleRequirement(title="Lead", seniority="manager", count=1, required_skills=[])
    engineer = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    role_rankings = {
        "Lead": [make_fit_score("a", "Lead", 90)],
        "Engineer": [make_fit_score("b", "Engineer", 90), make_fit_score("c", "Engineer", 80)],
    }

    order = _role_fill_order([engineer, lead], role_rankings)

    assert [r.title for r in order] == ["Lead", "Engineer"]


def test_assemble_team_skill_overlap_penalty_can_flip_the_winner():
    lead_role = RoleRequirement(title="Lead", seniority="manager", count=1, required_skills=[])
    engineer_role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[lead_role, engineer_role])

    lead1 = make_profile("lead1", skills=[make_skill("Python"), make_skill("Spark")])
    eng_overlap = make_profile("eng_overlap", skills=[make_skill("Python"), make_skill("Spark")])
    eng_complement = make_profile("eng_complement", skills=[make_skill("Java")])

    profiles_by_id = {p.consultant_id: p for p in [lead1, eng_overlap, eng_complement]}
    availability_by_id = {cid: availability_row(cid) for cid in profiles_by_id}
    role_rankings = {
        "Lead": [make_fit_score("lead1", "Lead", 90)],
        "Engineer": [
            make_fit_score("eng_overlap", "Engineer", 80),
            make_fit_score("eng_complement", "Engineer", 78),
        ],
    }

    team = assemble_team(brief, role_rankings, profiles_by_id, availability_by_id, graph={})

    engineer_member = next(m for m in team.members if m.role_title == "Engineer")
    assert engineer_member.consultant_id == "eng_complement"
    assert engineer_member.adjustments["skill_overlap_penalty"] == 0.0


def test_assemble_team_co_delivery_bonus_can_flip_the_winner():
    lead_role = RoleRequirement(title="Lead", seniority="manager", count=1, required_skills=[])
    engineer_role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[lead_role, engineer_role])

    lead1 = make_profile("lead1")
    eng_x = make_profile("eng_x")
    eng_y = make_profile("eng_y")

    profiles_by_id = {p.consultant_id: p for p in [lead1, eng_x, eng_y]}
    availability_by_id = {cid: availability_row(cid) for cid in profiles_by_id}
    role_rankings = {
        "Lead": [make_fit_score("lead1", "Lead", 90)],
        "Engineer": [
            make_fit_score("eng_y", "Engineer", 75),
            make_fit_score("eng_x", "Engineer", 70),
        ],
    }
    graph = {"eng_x": {"lead1": 2}, "lead1": {"eng_x": 2}}

    team = assemble_team(brief, role_rankings, profiles_by_id, availability_by_id, graph=graph)

    engineer_member = next(m for m in team.members if m.role_title == "Engineer")
    assert engineer_member.consultant_id == "eng_x"
    assert engineer_member.adjustments["co_delivery_bonus"] == 8.0


def test_assemble_team_booking_penalty_scales_with_missing_free_days():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])

    cand_a = make_profile("cand_a")
    cand_b = make_profile("cand_b")
    profiles_by_id = {p.consultant_id: p for p in [cand_a, cand_b]}
    availability_by_id = {
        "cand_a": availability_row("cand_a", status="available", free_days_per_week=5),
        "cand_b": availability_row("cand_b", status="partly_booked", free_days_per_week=1),
    }
    role_rankings = {"Engineer": [make_fit_score("cand_b", "Engineer", 95), make_fit_score("cand_a", "Engineer", 90)]}

    team = assemble_team(brief, role_rankings, profiles_by_id, availability_by_id, graph={})

    assert team.members[0].consultant_id == "cand_a"
    assert team.members[0].assembly_score == 90.0


def test_assemble_team_skips_role_when_no_ranked_candidates_remain():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])

    team = assemble_team(brief, {}, {}, {}, graph={})

    assert team.members == []
    assert len(team.gaps) == 1
    assert team.gaps[0].role_title == "Engineer"
    assert team.gaps[0].reason == "understaffed"


def test_assemble_team_flags_low_confidence_fit_as_a_gap():
    role = RoleRequirement(title="Quantum Cryptographer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])
    candidate = make_profile("cand")
    profiles_by_id = {"cand": candidate}
    availability_by_id = {"cand": availability_row("cand")}
    role_rankings = {"Quantum Cryptographer": [make_fit_score("cand", "Quantum Cryptographer", 6.0)]}

    team = assemble_team(brief, role_rankings, profiles_by_id, availability_by_id, graph={})

    assert len(team.gaps) == 1
    assert team.gaps[0].reason == "low_confidence_fit"
    assert team.gaps[0].consultant_id == "cand"


def test_assemble_team_no_gap_when_fit_score_meets_confidence_threshold():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])
    candidate = make_profile("cand")
    profiles_by_id = {"cand": candidate}
    availability_by_id = {"cand": availability_row("cand")}
    role_rankings = {"Engineer": [make_fit_score("cand", "Engineer", 75.0)]}

    team = assemble_team(brief, role_rankings, profiles_by_id, availability_by_id, graph={})

    assert team.gaps == []


def test_assemble_earliest_start_team_also_flags_low_confidence_fit():
    role = RoleRequirement(title="Quantum Cryptographer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])
    candidate = make_profile("cand")
    profiles_by_id = {"cand": candidate}
    availability_by_id = {"cand": availability_row("cand")}
    role_rankings = {"Quantum Cryptographer": [make_fit_score("cand", "Quantum Cryptographer", 6.0)]}

    team = assemble_earliest_start_team(brief, role_rankings, profiles_by_id, availability_by_id)

    assert len(team.gaps) == 1
    assert team.gaps[0].reason == "low_confidence_fit"


def test_staffing_gaps_reports_understaffed_when_count_not_met():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=2, required_skills=[])
    members = [
        TeamMember(
            role_title="Engineer",
            consultant_id="only_one",
            fit_score=80.0,
            reasons=REASONS,
            concern="None.",
            availability_status="available",
            free_days_per_week=5,
            next_free_date=TODAY.isoformat(),
            assembly_score=80.0,
        )
    ]

    gaps = _staffing_gaps([role], members)

    assert len(gaps) == 1
    assert gaps[0].reason == "understaffed"
    assert gaps[0].role_title == "Engineer"


def test_staffing_gaps_empty_when_role_fully_and_confidently_staffed():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    members = [
        TeamMember(
            role_title="Engineer",
            consultant_id="cand",
            fit_score=90.0,
            reasons=REASONS,
            concern="None.",
            availability_status="available",
            free_days_per_week=5,
            next_free_date=TODAY.isoformat(),
            assembly_score=90.0,
        )
    ]

    assert _staffing_gaps([role], members) == []


def test_assemble_earliest_start_team_prefers_soonest_availability_over_fit():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])

    high_fit_later = make_profile("high_fit_later")
    low_fit_sooner = make_profile("low_fit_sooner")
    profiles_by_id = {p.consultant_id: p for p in [high_fit_later, low_fit_sooner]}
    availability_by_id = {
        "high_fit_later": availability_row("high_fit_later", next_free_date="2026-06-01"),
        "low_fit_sooner": availability_row("low_fit_sooner", next_free_date="2026-04-01"),
    }
    role_rankings = {
        "Engineer": [
            make_fit_score("high_fit_later", "Engineer", 90),
            make_fit_score("low_fit_sooner", "Engineer", 70),
        ]
    }

    team = assemble_earliest_start_team(brief, role_rankings, profiles_by_id, availability_by_id)

    assert team.members[0].consultant_id == "low_fit_sooner"
    assert "2026-04-01" in team.members[0].selection_note


def test_assemble_lowest_cost_team_prefers_junior_seniority_over_fit():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])

    senior = make_profile("senior", seniority="principal")
    junior = make_profile("junior", seniority="analyst")
    profiles_by_id = {p.consultant_id: p for p in [senior, junior]}
    availability_by_id = {cid: availability_row(cid) for cid in profiles_by_id}
    role_rankings = {
        "Engineer": [make_fit_score("senior", "Engineer", 90), make_fit_score("junior", "Engineer", 70)]
    }

    team = assemble_lowest_cost_team(brief, role_rankings, profiles_by_id, availability_by_id)

    assert team.members[0].consultant_id == "junior"
    assert "analyst" in team.members[0].selection_note


def test_assemble_alt_team_only_considers_top_n_fit_scored_candidates():
    role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[role])

    # 6 candidates ranked by fit_score 100..50; the 6th (lowest fit) is earliest available
    # but should be excluded from the top-5 pool the alt team optimises within.
    fit_scores = [100, 90, 80, 70, 60, 50]
    dates = ["2026-05-05", "2026-05-04", "2026-05-03", "2026-05-02", "2026-05-01", "2020-01-01"]
    profiles_by_id = {}
    availability_by_id = {}
    rankings = []
    for i, (fit, next_free) in enumerate(zip(fit_scores, dates)):
        cid = f"c{i}"
        profiles_by_id[cid] = make_profile(cid)
        availability_by_id[cid] = availability_row(cid, next_free_date=next_free)
        rankings.append(make_fit_score(cid, "Engineer", fit))
    role_rankings = {"Engineer": rankings}

    team = assemble_earliest_start_team(brief, role_rankings, profiles_by_id, availability_by_id)

    assert team.members[0].consultant_id == "c4"  # earliest among the top-5 fit-scored, not the excluded c5


# --- Orchestrator ------------------------------------------------------------


def test_match_end_to_end_filters_reranks_and_assembles_three_teams():
    fits = make_profile(
        "fits",
        skills=[make_skill("Python")],
        languages=[make_language("English")],
        location="Copenhagen, Denmark",
    )
    missing_skill = make_profile(
        "missing_skill",
        skills=[make_skill("Java")],
        languages=[make_language("English")],
        location="Copenhagen, Denmark",
    )
    booked_but_qualified = make_profile(
        "booked_but_qualified",
        skills=[make_skill("Python")],
        languages=[make_language("English")],
        location="Copenhagen, Denmark",
    )
    role = RoleRequirement(title="Dev", seniority="consultant", count=1, required_skills=["Python"])
    brief = make_brief(
        roles_needed=[role],
        must_have_skills=["Python"],
        required_language="English",
        location="Copenhagen",
        start_date=TODAY.isoformat(),
    )
    availability = [
        availability_row("fits"),
        availability_row("missing_skill"),
        availability_row("booked_but_qualified", status="fully_booked", next_free_date="2026-03-20"),
    ]
    client = _FakeClient(
        [{"rankings": [{"consultant_id": "fits", "fit_score": 85, "reasons": REASONS, "concern": "None major."}]}]
    )
    embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)  # noqa: E731

    result = match(
        [fits, missing_skill, booked_but_qualified],
        brief,
        availability,
        graph={},
        client=client,
        embed_fn=embed_fn,
        today=TODAY,
    )

    stages = {stage.stage: stage.survived for stage in result.funnel}
    assert stages["total"] == 3
    assert stages["availability"] == 2  # booked_but_qualified dropped here, before must_have_skills is even checked
    assert stages["must_have_skills"] == 1  # of the 2 remaining, missing_skill drops here

    assert [t.consultant_id for t in result.availability_tradeoffs] == ["booked_but_qualified"]
    assert result.availability_tradeoffs[0].days_after_start == 19

    assert len(result.teams) == 3
    assert {team.label for team in result.teams} == {"recommended", "earliest_start", "lowest_cost"}
    for team in result.teams:
        assert [m.consultant_id for m in team.members] == ["fits"]


def test_match_reranks_multiple_roles_concurrently_and_assembles_both():
    lead = make_profile("lead1")
    engineer = make_profile("eng1")
    lead_role = RoleRequirement(title="Lead", seniority="manager", count=1, required_skills=[])
    engineer_role = RoleRequirement(title="Engineer", seniority="consultant", count=1, required_skills=[])
    brief = make_brief(roles_needed=[lead_role, engineer_role])
    availability = [availability_row("lead1"), availability_row("eng1")]
    client = _RoleAwareFakeClient(
        {
            "Lead": {"rankings": [{"consultant_id": "lead1", "fit_score": 90, "reasons": REASONS, "concern": "None."}]},
            "Engineer": {"rankings": [{"consultant_id": "eng1", "fit_score": 70, "reasons": REASONS, "concern": "None."}]},
        }
    )
    embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)  # noqa: E731

    result = match([lead, engineer], brief, availability, graph={}, client=client, embed_fn=embed_fn, today=TODAY)

    assert len(client.messages.calls) == 2  # one rerank_role call per role, dispatched concurrently
    recommended = next(t for t in result.teams if t.label == "recommended")
    assert {m.consultant_id for m in recommended.members} == {"lead1", "eng1"}
    fit_by_id = {m.consultant_id: m.fit_score for m in recommended.members}
    assert fit_by_id == {"lead1": 90.0, "eng1": 70.0}
