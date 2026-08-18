from src.explain import (
    _availability_score,
    _future_fit_rank,
    _industry_experience,
    _seniority_fit,
    _skill_fit,
    _summarize_availability_alternative,
    _summarize_swap,
    _team_chemistry,
    build_availability_alternative,
    build_counterfactual,
    build_match_card,
    build_match_cards_for_team,
    classify_trust,
    compute_breakdown,
    is_confirmed_trust_finding,
    select_evidence,
)
from src.schema import (
    AvailabilityTradeoff,
    ConsultantProfile,
    ProjectBrief,
    RoleFitScore,
    RoleRequirement,
    Team,
    TeamMember,
)

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


def make_skill(name, confidence=0.9, evidence=None, category="technical"):
    return {
        "name": name,
        "category": category,
        "evidence": evidence or f"Used {name} extensively on client engagements.",
        "confidence": confidence,
    }


def make_project(title, tech=None, role="Consultant", impact="Delivered on time."):
    return {
        "title": title,
        "industry": "Financial Services",
        "role": role,
        "impact": impact,
        "tech": tech or [],
        "year_start": 2023,
    }


def make_brief(**overrides):
    defaults = dict(client="Acme", industry="Retail", roles_needed=[])
    return ProjectBrief(**{**defaults, **overrides})


def make_role(**overrides):
    defaults = dict(title="Engineer", seniority="consultant", count=1, required_skills=[])
    return RoleRequirement(**{**defaults, **overrides})


def make_fit_score(consultant_id, role_title, fit_score, reasons=None, concern="A minor gap."):
    return RoleFitScore(
        consultant_id=consultant_id,
        role_title=role_title,
        fit_score=fit_score,
        reasons=reasons or REASONS,
        concern=concern,
    )


def make_member(consultant_id, role_title="Engineer", fit_score=80.0, status="available", free_days=5):
    return TeamMember(
        role_title=role_title,
        consultant_id=consultant_id,
        fit_score=fit_score,
        reasons=REASONS,
        concern="A minor gap.",
        availability_status=status,
        free_days_per_week=free_days,
        next_free_date="2026-08-17",
        assembly_score=fit_score,
    )


def availability_row(consultant_id, status="available", free_days_per_week=5, next_free_date="2026-08-17"):
    return {
        "consultant_id": consultant_id,
        "status": status,
        "free_days_per_week": free_days_per_week,
        "next_free_date": next_free_date,
        "current_project": "",
    }


def make_tradeoff(consultant_id, next_free_date="2026-09-14", days_after_start=13):
    return AvailabilityTradeoff(consultant_id=consultant_id, next_free_date=next_free_date, days_after_start=days_after_start)


# --- score breakdown components -------------------------------------------


def test_skill_fit_full_match_scores_100():
    profile = make_profile("p1", skills=[make_skill("Python"), make_skill("SQL")])
    role = make_role(required_skills=["Python", "SQL"])
    assert _skill_fit(profile, role, make_brief()) == 100.0


def test_skill_fit_partial_match_scores_proportionally():
    profile = make_profile("p1", skills=[make_skill("Python")])
    role = make_role(required_skills=["Python", "SQL", "Docker", "Kubernetes"])
    assert _skill_fit(profile, role, make_brief()) == 25.0


def test_skill_fit_no_requirements_scores_100():
    profile = make_profile("p1", skills=[])
    role = make_role(required_skills=[])
    assert _skill_fit(profile, role, make_brief(must_have_skills=[])) == 100.0


def test_skill_fit_includes_brief_must_have_skills():
    profile = make_profile("p1", skills=[make_skill("Python")])
    role = make_role(required_skills=[])
    assert _skill_fit(profile, role, make_brief(must_have_skills=["Python", "SQL"])) == 50.0


def test_seniority_fit_exact_match_scores_100():
    profile = make_profile("p1", seniority="senior_consultant")
    role = make_role(seniority="senior_consultant")
    assert _seniority_fit(profile, role) == 100.0


def test_seniority_fit_penalizes_distance_symmetrically():
    role = make_role(seniority="senior_consultant")
    one_tier_off = make_profile("p1", seniority="consultant")
    two_tiers_off = make_profile("p2", seniority="analyst")
    assert _seniority_fit(one_tier_off, role) == 80.0
    assert _seniority_fit(two_tiers_off, role) == 60.0


def test_availability_score_by_status():
    assert _availability_score("available", 5) == 100.0
    assert _availability_score("partly_booked", 0) == 50.0
    assert _availability_score("partly_booked", 5) == 100.0
    assert _availability_score("fully_booked", 0) == 60.0


def test_industry_experience_matches_by_substring():
    profile = make_profile("p1", industries=["Financial Services"])
    assert _industry_experience(profile, make_brief(industry="Financial Services")) == 80.0
    assert _industry_experience(profile, make_brief(industry="Retail")) == 25.0


def test_industry_experience_no_brief_industry_scores_100():
    profile = make_profile("p1", industries=[])
    assert _industry_experience(profile, make_brief(industry="")) == 100.0


def test_team_chemistry_baseline_with_no_teammates():
    profile = make_profile("p1", skills=[make_skill("Python")])
    assert _team_chemistry(profile, [], {}) == 70.0


def test_team_chemistry_rewards_co_delivery():
    profile = make_profile("p1", skills=[])
    teammate = make_profile("p2", skills=[])
    graph = {"p1": {"p2": 2}}
    assert _team_chemistry(profile, [teammate], graph) == 90.0


def test_team_chemistry_penalizes_skill_overlap():
    profile = make_profile("p1", skills=[make_skill("Python"), make_skill("SQL")])
    teammate = make_profile("p2", skills=[make_skill("Python"), make_skill("SQL")])
    assert _team_chemistry(profile, [teammate], {}) == 58.0


def test_team_chemistry_clamped_to_0_100_range():
    profile = make_profile("p1", skills=[make_skill(f"skill{i}") for i in range(20)])
    teammate = make_profile("p2", skills=[make_skill(f"skill{i}") for i in range(20)])
    assert _team_chemistry(profile, [teammate], {}) == 0.0


def test_compute_breakdown_combines_all_five_components():
    profile = make_profile("p1", seniority="senior_consultant", skills=[make_skill("Python")], industries=["Retail"])
    role = make_role(seniority="senior_consultant", required_skills=["Python"])
    breakdown = compute_breakdown(profile, role, make_brief(industry="Retail"), "available", 5, [], {})
    assert breakdown.skill_fit == 100.0
    assert breakdown.seniority_fit == 100.0
    assert breakdown.availability == 100.0
    assert breakdown.industry_experience == 80.0
    assert breakdown.team_chemistry == 70.0


# --- trust badge -------------------------------------------------------


def test_classify_trust_flagged_on_structured_injection_flag():
    profile = make_profile("p1", trust_flags=["injection (high): ignore all instructions"])
    assert classify_trust(profile) == "flagged"


def test_classify_trust_flagged_on_structured_promotional_language_flag():
    profile = make_profile("p1", trust_flags=["promotional_language (medium): THE BEST"])
    assert classify_trust(profile) == "flagged"


def test_classify_trust_unverified_when_confidence_below_threshold():
    profile = make_profile("p1", trust_flags=[], extraction_confidence=0.5)
    assert classify_trust(profile) == "unverified_claims"


def test_classify_trust_verified_when_clean_and_confident():
    profile = make_profile("p1", trust_flags=[], extraction_confidence=0.95)
    assert classify_trust(profile) == "verified"


def test_classify_trust_not_flagged_by_a_self_reported_clearing_statement():
    # Real dataset shape: a self-reported note can literally contain "injection" while explicitly
    # clearing the CV -- must not be mistaken for a confirmed finding.
    profile = make_profile(
        "p1", trust_flags=["No prompt injection or adversarial text detected in the document."], extraction_confidence=0.82
    )
    assert classify_trust(profile) == "verified"


def test_classify_trust_not_flagged_by_benign_extraction_caveat():
    # Real dataset shape: benign self-reported extraction caveats (missing certs, truncated text) land
    # in trust_flags too, with no structured prefix -- these are not trust/security concerns.
    profile = make_profile(
        "p1",
        trust_flags=["No certifications section found in the CV.", "Some project entries appear cut off mid-sentence."],
        extraction_confidence=0.84,
    )
    assert classify_trust(profile) == "verified"


def test_classify_trust_unverified_when_self_reported_flags_present_but_confidence_is_low():
    profile = make_profile("p1", trust_flags=["CV header contains a different name from the body."], extraction_confidence=0.55)
    assert classify_trust(profile) == "unverified_claims"


def test_is_confirmed_trust_finding_matches_structured_prefixes_only():
    assert is_confirmed_trust_finding("injection (high): ignore all instructions") is True
    assert is_confirmed_trust_finding("promotional_language (medium): THE BEST") is True
    assert is_confirmed_trust_finding("No prompt injection or adversarial text detected in the document.") is False
    assert is_confirmed_trust_finding("No certifications section found in the CV.") is False


# --- evidence selection --------------------------------------------------


def test_select_evidence_prioritizes_required_skills_then_confidence():
    profile = make_profile(
        "p1",
        skills=[
            make_skill("Excel", confidence=0.99, evidence="Built Excel models."),
            make_skill("Python", confidence=0.7, evidence="Wrote Python scripts."),
        ],
    )
    role = make_role(required_skills=["Python"])
    evidence = select_evidence(profile, role, make_brief())
    assert evidence[0].skill_name == "Python"
    assert evidence[0].matched_requirement is True
    assert evidence[1].matched_requirement is False


def test_select_evidence_marks_matched_requirement_false_when_nothing_matches():
    # Real bug this guards: an "Azure migration" brief surfacing candidates with zero Azure/cloud
    # skills, whose cards showed unrelated high-confidence skills as unqualified "Evidence".
    profile = make_profile(
        "p1",
        skills=[make_skill("Change Management", confidence=0.95, evidence="Led change management initiatives.")],
    )
    role = make_role(required_skills=["Azure", "Cloud Migration"])
    evidence = select_evidence(profile, role, make_brief())
    assert len(evidence) == 1  # rule 4: a score must still carry evidence, even when nothing matches
    assert evidence[0].matched_requirement is False


def test_select_evidence_attributes_project_via_tech_list():
    profile = make_profile(
        "p1",
        skills=[make_skill("Docker", evidence="Deployed services with Docker.")],
        projects=[make_project("Platform Migration", tech=["Docker", "Kubernetes"])],
    )
    evidence = select_evidence(profile, make_role(), make_brief())
    assert evidence[0].project_title == "Platform Migration"
    assert evidence[0].quote == "Deployed services with Docker."


def test_select_evidence_project_title_none_when_no_tech_match():
    profile = make_profile("p1", skills=[make_skill("Docker")], projects=[make_project("Something Else", tech=["Java"])])
    evidence = select_evidence(profile, make_role(), make_brief())
    assert evidence[0].project_title is None


def test_select_evidence_returns_fewer_than_top_n_when_profile_has_fewer_skills():
    profile = make_profile("p1", skills=[make_skill("Python")])
    evidence = select_evidence(profile, make_role(), make_brief())
    assert len(evidence) == 1


def test_select_evidence_skips_duplicate_evidence_text():
    shared_quote = "Python, SQL, Spark, TensorFlow, Docker"
    profile = make_profile(
        "p1",
        skills=[
            make_skill("Python", confidence=0.95, evidence=shared_quote),
            make_skill("SQL", confidence=0.9, evidence=shared_quote),
            make_skill("Spark", confidence=0.85, evidence=shared_quote),
            make_skill("Predictive Modeling", confidence=0.8, evidence="Led a predictive modeling initiative."),
        ],
    )
    evidence = select_evidence(profile, make_role(), make_brief())
    assert len(evidence) == 2  # only 2 distinct evidence strings exist across all 4 skills
    assert evidence[0].quote == shared_quote
    assert evidence[1].quote == "Led a predictive modeling initiative."


# --- counterfactual summary templating -----------------------------------


def test_summarize_swap_gain_and_loss():
    deltas = {"skill_fit": 20.0, "seniority_fit": -10.0, "availability": 0.0, "industry_experience": 0.0, "team_chemistry": 0.0}
    summary = _summarize_swap("alt1", deltas, "verified")
    assert "improve skill fit (+20)" in summary
    assert "reduce seniority fit (-10)" in summary


def test_summarize_swap_no_downside_when_all_gains():
    deltas = {"skill_fit": 20.0, "seniority_fit": 5.0, "availability": 0.0, "industry_experience": 0.0, "team_chemistry": 0.0}
    summary = _summarize_swap("alt1", deltas, "verified")
    assert "no clear downside" in summary
    assert "skill fit" in summary


def test_summarize_swap_strict_downgrade_when_all_losses():
    deltas = {"skill_fit": -20.0, "seniority_fit": -5.0, "availability": 0.0, "industry_experience": 0.0, "team_chemistry": 0.0}
    summary = _summarize_swap("alt1", deltas, "verified")
    assert "strict downgrade" in summary
    assert "skill fit" in summary


def test_summarize_swap_appends_trust_caveat_when_flagged():
    deltas = {"skill_fit": 20.0, "seniority_fit": 5.0, "availability": 0.0, "industry_experience": 0.0, "team_chemistry": 0.0}
    summary = _summarize_swap("alt1", deltas, "flagged")
    assert "no clear downside" in summary
    assert "flagged trust issue" in summary


def test_summarize_swap_no_trust_caveat_when_verified():
    deltas = {"skill_fit": 20.0, "seniority_fit": 5.0, "availability": 0.0, "industry_experience": 0.0, "team_chemistry": 0.0}
    summary = _summarize_swap("alt1", deltas, "verified")
    assert "Note:" not in summary


# --- build_counterfactual --------------------------------------------------


def test_build_counterfactual_returns_none_when_no_alternatives():
    role = make_role()
    result = build_counterfactual([], "selected", set(), role, make_brief(), {}, {}, [], {}, compute_breakdown(
        make_profile("selected"), role, make_brief(), "available", 5, [], {}
    ))
    assert result is None


def test_build_counterfactual_excludes_candidates_already_on_team():
    role = make_role()
    brief = make_brief()
    selected = make_profile("selected")
    on_team_elsewhere = make_profile("on_team")
    remaining_alt = make_profile("remaining_alt")
    profiles_by_id = {"selected": selected, "on_team": on_team_elsewhere, "remaining_alt": remaining_alt}
    availability_by_id = {
        "on_team": availability_row("on_team"),
        "remaining_alt": availability_row("remaining_alt"),
    }
    role_ranking = [
        make_fit_score("selected", role.title, 90),
        make_fit_score("on_team", role.title, 95),  # higher fit, but excluded -- already staffed elsewhere
        make_fit_score("remaining_alt", role.title, 70),
    ]
    selected_breakdown = compute_breakdown(selected, role, brief, "available", 5, [], {})

    result = build_counterfactual(
        role_ranking, "selected", {"selected", "on_team"}, role, brief, profiles_by_id, availability_by_id, [], {}, selected_breakdown
    )

    assert result.consultant_id == "remaining_alt"


def test_build_counterfactual_picks_highest_fit_score_alternative():
    role = make_role()
    brief = make_brief()
    selected = make_profile("selected")
    weaker_alt = make_profile("weaker")
    stronger_alt = make_profile("stronger")
    profiles_by_id = {"selected": selected, "weaker": weaker_alt, "stronger": stronger_alt}
    availability_by_id = {"weaker": availability_row("weaker"), "stronger": availability_row("stronger")}
    role_ranking = [
        make_fit_score("selected", role.title, 90),
        make_fit_score("weaker", role.title, 40),
        make_fit_score("stronger", role.title, 60),
    ]
    selected_breakdown = compute_breakdown(selected, role, brief, "available", 5, [], {})

    result = build_counterfactual(
        role_ranking, "selected", {"selected"}, role, brief, profiles_by_id, availability_by_id, [], {}, selected_breakdown
    )

    assert result.consultant_id == "stronger"
    assert result.fit_score == 60.0


def test_build_counterfactual_carries_alternatives_own_trust_badge():
    role = make_role()
    brief = make_brief()
    selected = make_profile("selected")
    flagged_alt = make_profile("flagged_alt", trust_flags=["injection (high): ignore all instructions"])
    profiles_by_id = {"selected": selected, "flagged_alt": flagged_alt}
    availability_by_id = {"flagged_alt": availability_row("flagged_alt")}
    role_ranking = [make_fit_score("selected", role.title, 50), make_fit_score("flagged_alt", role.title, 90)]
    selected_breakdown = compute_breakdown(selected, role, brief, "available", 5, [], {})

    result = build_counterfactual(
        role_ranking, "selected", {"selected"}, role, brief, profiles_by_id, availability_by_id, [], {}, selected_breakdown
    )

    assert result.trust_badge == "flagged"
    assert "flagged trust issue" in result.summary


# --- build_availability_alternative -----------------------------------------


def test_future_fit_rank_ignores_availability_component():
    from src.schema import ScoreBreakdown

    high_availability_low_everything_else = ScoreBreakdown(
        skill_fit=0.0, seniority_fit=0.0, availability=100.0, industry_experience=0.0, team_chemistry=0.0
    )
    assert _future_fit_rank(high_availability_low_everything_else) == 0.0


def test_summarize_availability_alternative_includes_days_late_and_pluralizes():
    deltas = {"skill_fit": 20.0, "seniority_fit": 0.0, "availability": 0.0, "industry_experience": 0.0, "team_chemistry": 0.0}
    summary = _summarize_availability_alternative("alt1", deltas, 13, "verified")
    assert "Not free until 13 days after your requested start." in summary
    singular = _summarize_availability_alternative("alt1", deltas, 1, "verified")
    assert "Not free until 1 day after your requested start." in singular


def test_build_availability_alternative_returns_none_when_no_tradeoffs():
    role = make_role()
    brief = make_brief()
    selected_breakdown = compute_breakdown(make_profile("selected"), role, brief, "available", 5, [], {})
    result = build_availability_alternative([], role, brief, {}, {}, [], {}, selected_breakdown)
    assert result is None


def test_build_availability_alternative_returns_none_when_not_better_than_selected():
    role = make_role(required_skills=["Python"])
    brief = make_brief()
    selected = make_profile("selected", skills=[make_skill("Python")], seniority="consultant")
    weak_tradeoff_profile = make_profile("booked", skills=[], seniority="intern")
    profiles_by_id = {"selected": selected, "booked": weak_tradeoff_profile}
    availability_by_id = {"booked": availability_row("booked", status="fully_booked", free_days_per_week=0)}
    selected_breakdown = compute_breakdown(selected, role, brief, "available", 5, [], {})

    result = build_availability_alternative(
        [make_tradeoff("booked")], role, brief, profiles_by_id, availability_by_id, [], {}, selected_breakdown
    )

    assert result is None


def test_build_availability_alternative_surfaces_a_genuinely_stronger_candidate():
    role = make_role(required_skills=["Python", "SQL"], seniority="senior_consultant")
    brief = make_brief(industry="Retail")
    selected = make_profile("selected", skills=[], seniority="intern", industries=[])
    strong_tradeoff_profile = make_profile(
        "booked", skills=[make_skill("Python"), make_skill("SQL")], seniority="senior_consultant", industries=["Retail"]
    )
    profiles_by_id = {"selected": selected, "booked": strong_tradeoff_profile}
    availability_by_id = {"booked": availability_row("booked", status="fully_booked", free_days_per_week=0)}
    selected_breakdown = compute_breakdown(selected, role, brief, "available", 5, [], {})

    result = build_availability_alternative(
        [make_tradeoff("booked", next_free_date="2026-09-14", days_after_start=13)],
        role,
        brief,
        profiles_by_id,
        availability_by_id,
        [],
        {},
        selected_breakdown,
    )

    assert result is not None
    assert result.consultant_id == "booked"
    assert result.days_after_start == 13
    assert result.next_free_date == "2026-09-14"
    assert "Not free until 13 days after your requested start." in result.summary


def test_build_availability_alternative_picks_the_strongest_of_multiple_tradeoffs():
    role = make_role(required_skills=["Python"])
    brief = make_brief()
    selected = make_profile("selected", skills=[], seniority="intern")
    weaker = make_profile("weaker", skills=[], seniority="intern")
    stronger = make_profile("stronger", skills=[make_skill("Python")], seniority="senior_consultant")
    profiles_by_id = {"selected": selected, "weaker": weaker, "stronger": stronger}
    availability_by_id = {
        "weaker": availability_row("weaker", status="fully_booked", free_days_per_week=0),
        "stronger": availability_row("stronger", status="fully_booked", free_days_per_week=0),
    }
    selected_breakdown = compute_breakdown(selected, role, brief, "available", 5, [], {})

    result = build_availability_alternative(
        [make_tradeoff("weaker"), make_tradeoff("stronger")],
        role,
        brief,
        profiles_by_id,
        availability_by_id,
        [],
        {},
        selected_breakdown,
    )

    assert result.consultant_id == "stronger"


def test_build_availability_alternative_carries_trust_badge():
    role = make_role(required_skills=["Python"])
    brief = make_brief()
    selected = make_profile("selected", skills=[], seniority="intern")
    flagged = make_profile(
        "booked", skills=[make_skill("Python")], seniority="senior_consultant", trust_flags=["injection (high): x"]
    )
    profiles_by_id = {"selected": selected, "booked": flagged}
    availability_by_id = {"booked": availability_row("booked", status="fully_booked", free_days_per_week=0)}
    selected_breakdown = compute_breakdown(selected, role, brief, "available", 5, [], {})

    result = build_availability_alternative(
        [make_tradeoff("booked")], role, brief, profiles_by_id, availability_by_id, [], {}, selected_breakdown
    )

    assert result.trust_badge == "flagged"
    assert "flagged trust issue" in result.summary


# --- build_match_card / build_match_cards_for_team --------------------------


def test_build_match_card_end_to_end():
    role = make_role(title="Engineer", seniority="consultant", required_skills=["Python"])
    brief = make_brief(industry="Retail", roles_needed=[role])
    selected_profile = make_profile(
        "selected", skills=[make_skill("Python", evidence="Wrote Python services.")], industries=["Retail"]
    )
    alt_profile = make_profile("alt", skills=[make_skill("Python", confidence=0.99, evidence="Led Python migration.")])
    profiles_by_id = {"selected": selected_profile, "alt": alt_profile}
    availability_by_id = {"alt": availability_row("alt")}

    member = make_member("selected", role_title="Engineer", fit_score=75.0)
    team = Team(label="recommended", members=[member])
    role_rankings = {"Engineer": [make_fit_score("selected", "Engineer", 75), make_fit_score("alt", "Engineer", 85)]}

    card = build_match_card(member, team, brief, role_rankings, profiles_by_id, availability_by_id, {})

    assert card.consultant_id == "selected"
    assert card.role_title == "Engineer"
    assert card.overall_score == 75.0
    assert card.trust_badge == "verified"
    assert len(card.evidence) == 1
    assert card.evidence[0].quote == "Wrote Python services."
    assert card.counterfactual is not None
    assert card.counterfactual.consultant_id == "alt"
    assert card.availability_alternative is None  # no tradeoffs passed in


def test_build_match_card_wires_through_availability_alternative():
    role = make_role(title="Engineer", seniority="senior_consultant", required_skills=["Python"])
    brief = make_brief(roles_needed=[role])
    selected_profile = make_profile("selected", skills=[], seniority="intern")
    booked_profile = make_profile("booked", skills=[make_skill("Python")], seniority="senior_consultant")
    profiles_by_id = {"selected": selected_profile, "booked": booked_profile}
    availability_by_id = {"booked": availability_row("booked", status="fully_booked", free_days_per_week=0)}

    member = make_member("selected", role_title="Engineer", fit_score=40.0)
    team = Team(label="recommended", members=[member])
    tradeoffs = [make_tradeoff("booked", next_free_date="2026-09-14", days_after_start=13)]

    card = build_match_card(member, team, brief, {}, profiles_by_id, availability_by_id, {}, tradeoffs)

    assert card.availability_alternative is not None
    assert card.availability_alternative.consultant_id == "booked"
    assert card.availability_alternative.days_after_start == 13


def test_build_match_cards_for_team_returns_one_card_per_member():
    role_a = make_role(title="Lead", seniority="manager", required_skills=[])
    role_b = make_role(title="Engineer", seniority="consultant", required_skills=[])
    brief = make_brief(roles_needed=[role_a, role_b])
    lead_profile = make_profile("lead1")
    eng_profile = make_profile("eng1")
    profiles_by_id = {"lead1": lead_profile, "eng1": eng_profile}
    availability_by_id = {"lead1": availability_row("lead1"), "eng1": availability_row("eng1")}

    lead_member = make_member("lead1", role_title="Lead", fit_score=90.0)
    eng_member = make_member("eng1", role_title="Engineer", fit_score=70.0)
    team = Team(label="recommended", members=[lead_member, eng_member])
    role_rankings = {
        "Lead": [make_fit_score("lead1", "Lead", 90)],
        "Engineer": [make_fit_score("eng1", "Engineer", 70)],
    }

    cards = build_match_cards_for_team(team, brief, role_rankings, profiles_by_id, availability_by_id, {})

    assert [c.consultant_id for c in cards] == ["lead1", "eng1"]
    assert all(c.counterfactual is None for c in cards)  # each role only has the one ranked candidate
