from app import (
    _EXAMPLE_BRIEFS,
    availability_rows,
    brief_chips,
    format_elapsed,
    funnel_rows,
    graph_edges,
    network_layout,
    role_shortlist_sizes,
    split_trust_flags,
)
from src.schema import ConsultantProfile, FilterFunnelStage, MatchResult, ProjectBrief, RoleFitScore, RoleRequirement

BASE_PROFILE = dict(
    current_role="Consultant",
    seniority="consultant",
    years_experience=5,
    location="Copenhagen, Denmark",
    extraction_confidence=0.9,
)


def make_profile(consultant_id, **overrides):
    return ConsultantProfile(**{**BASE_PROFILE, "consultant_id": consultant_id, **overrides})


def make_brief(**overrides):
    defaults = dict(client="Acme", industry="Retail", roles_needed=[])
    return ProjectBrief(**{**defaults, **overrides})


def make_result(**overrides):
    defaults = dict(funnel=[], teams=[])
    return MatchResult(**{**defaults, **overrides})


# --- example briefs ----------------------------------------------------


def test_example_briefs_has_exactly_three_distinct_non_empty_prompts():
    assert len(_EXAMPLE_BRIEFS) == 3
    texts = list(_EXAMPLE_BRIEFS.values())
    assert len(set(texts)) == 3
    assert all(text.strip() for text in texts)


# --- brief_chips ---------------------------------------------------------


def test_brief_chips_marks_inferred_fields():
    brief = make_brief(client="Acme", industry="Retail", inferred_fields=["client", "budget"])
    rows = {row["label"]: row for row in brief_chips(brief)}
    assert rows["Client"]["inferred"] is True
    assert rows["Industry"]["inferred"] is False
    assert rows["Budget"]["inferred"] is True


def test_brief_chips_shows_not_specified_for_missing_optional_fields():
    brief = make_brief(start_date=None, duration_weeks=None, location=None, required_language=None, budget=None)
    rows = {row["label"]: row for row in brief_chips(brief)}
    assert rows["Start date"]["value"] == "Not specified"
    assert rows["Duration"]["value"] == "Not specified"
    assert rows["Location"]["value"] == "Not specified"
    assert rows["Required language"]["value"] == "Not specified"
    assert rows["Budget"]["value"] == "Not disclosed"


def test_brief_chips_formats_duration_and_skills():
    brief = make_brief(duration_weeks=24, must_have_skills=["Python", "SQL"])
    rows = {row["label"]: row for row in brief_chips(brief)}
    assert rows["Duration"]["value"] == "24 weeks"
    assert rows["Must-have skills"]["value"] == "Python, SQL"


# --- funnel_rows / role_shortlist_sizes -----------------------------------


def test_funnel_rows_formats_stage_labels_and_preserves_order():
    result = make_result(
        funnel=[
            FilterFunnelStage(stage="total", survived=21),
            FilterFunnelStage(stage="required_language", survived=15),
        ]
    )
    assert funnel_rows(result) == [("Total", 21), ("Required Language", 15)]


def test_role_shortlist_sizes_counts_ranked_candidates_per_role():
    result = make_result(
        role_rankings={
            "Lead": [RoleFitScore(consultant_id="a", role_title="Lead", fit_score=90, reasons=["x", "y", "z"], concern="c")],
            "Engineer": [
                RoleFitScore(consultant_id="a", role_title="Engineer", fit_score=80, reasons=["x", "y", "z"], concern="c"),
                RoleFitScore(consultant_id="b", role_title="Engineer", fit_score=70, reasons=["x", "y", "z"], concern="c"),
            ],
        }
    )
    assert role_shortlist_sizes(result) == {"Lead": 1, "Engineer": 2}


# --- format_elapsed --------------------------------------------------------


def test_format_elapsed_under_a_minute_shows_seconds():
    assert format_elapsed(12.3) == "12.3s"


def test_format_elapsed_over_a_minute_shows_minutes_and_seconds():
    assert format_elapsed(76.0) == "1m 16s"


def test_format_elapsed_exactly_zero():
    assert format_elapsed(0.0) == "0.0s"


# --- split_trust_flags -----------------------------------------------------


def test_split_trust_flags_separates_confirmed_findings_from_extraction_notes():
    injected = make_profile("cv4", trust_flags=["injection (high): ignore all instructions"])
    clean_note = make_profile("cv_7", trust_flags=["No certifications section found in the CV."])
    clean = make_profile("cv1", trust_flags=[])

    confirmed, notes = split_trust_flags([injected, clean_note, clean])

    assert confirmed == [{"consultant_id": "cv4", "flag": "injection (high): ignore all instructions"}]
    assert notes == [{"consultant_id": "cv_7", "flag": "No certifications section found in the CV."}]


def test_split_trust_flags_empty_when_no_profiles_have_flags():
    confirmed, notes = split_trust_flags([make_profile("cv1"), make_profile("cv2")])
    assert confirmed == []
    assert notes == []


# --- graph_edges ------------------------------------------------------------


def test_graph_edges_dedupes_both_directions_into_one_row():
    graph = {"a": {"b": 2}, "b": {"a": 2}}
    edges = graph_edges(graph)
    assert len(edges) == 1
    assert edges[0]["shared_projects"] == 2
    assert {edges[0]["a"], edges[0]["b"]} == {"a", "b"}


def test_graph_edges_sorted_by_weight_descending():
    graph = {"a": {"b": 1}, "b": {"a": 1, "c": 5}, "c": {"b": 5}}
    edges = graph_edges(graph)
    assert [e["shared_projects"] for e in edges] == [5, 1]


def test_graph_edges_empty_graph():
    assert graph_edges({}) == []


# --- availability_rows -------------------------------------------------------


def test_availability_rows_sorted_available_before_partly_before_fully_booked():
    profiles = [make_profile("fully"), make_profile("available"), make_profile("partly")]
    availability_by_id = {
        "fully": {"status": "fully_booked", "free_days_per_week": 0, "next_free_date": "2026-09-01"},
        "available": {"status": "available", "free_days_per_week": 5, "next_free_date": "2026-08-19"},
        "partly": {"status": "partly_booked", "free_days_per_week": 2, "next_free_date": "2026-08-19"},
    }
    rows = availability_rows(profiles, availability_by_id, {})
    assert [r["consultant_id"] for r in rows] == ["available", "partly", "fully"]


def test_availability_rows_uses_display_name_when_available():
    profiles = [make_profile("cv1")]
    availability_by_id = {"cv1": {"status": "available", "free_days_per_week": 5, "next_free_date": "2026-08-19"}}
    rows = availability_rows(profiles, availability_by_id, {"cv1": "Maria Lund"})
    assert rows[0]["name"] == "Maria Lund"


def test_availability_rows_skips_profiles_with_no_availability_data():
    profiles = [make_profile("cv1"), make_profile("cv2")]
    availability_by_id = {"cv1": {"status": "available", "free_days_per_week": 5, "next_free_date": "2026-08-19"}}
    rows = availability_rows(profiles, availability_by_id, {})
    assert [r["consultant_id"] for r in rows] == ["cv1"]


# --- network_layout ----------------------------------------------------------


def test_network_layout_excludes_isolated_nodes():
    graph = {"a": {"b": 1}, "b": {"a": 1}, "c": {}}
    layout = network_layout(graph, {})
    assert {n["id"] for n in layout["nodes"]} == {"a", "b"}


def test_network_layout_one_edge_per_pair_not_two():
    graph = {"a": {"b": 3}, "b": {"a": 3}}
    layout = network_layout(graph, {})
    assert len(layout["edges"]) == 1
    assert layout["edges"][0]["weight"] == 3


def test_network_layout_nodes_use_display_names():
    graph = {"a": {"b": 1}, "b": {"a": 1}}
    layout = network_layout(graph, {"a": "Maria Lund", "b": "Jonas Mikkelsen"})
    names = {n["id"]: n["name"] for n in layout["nodes"]}
    assert names == {"a": "Maria Lund", "b": "Jonas Mikkelsen"}


def test_network_layout_positions_nodes_within_the_svg_bounds():
    graph = {"a": {"b": 1}, "b": {"a": 1, "c": 2}, "c": {"b": 2}}
    layout = network_layout(graph, {}, size=400.0)
    for node in layout["nodes"]:
        assert 0 <= node["x"] <= 400
        assert 0 <= node["y"] <= 400


def test_network_layout_empty_graph_returns_no_nodes_or_edges():
    layout = network_layout({}, {})
    assert layout["nodes"] == []
    assert layout["edges"] == []
