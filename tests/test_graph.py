from src.graph import build_co_delivery_graph
from src.schema import ConsultantProfile

BASE = dict(
    current_role="Consultant",
    seniority="consultant",
    years_experience=5,
    location="Copenhagen, Denmark",
    extraction_confidence=0.9,
)


def make_profile(consultant_id, project_titles):
    return ConsultantProfile(
        **BASE,
        consultant_id=consultant_id,
        projects=[
            {
                "title": title,
                "industry": "Financial Services",
                "role": "Consultant",
                "impact": "Delivered on time.",
                "tech": [],
                "year_start": 2022,
            }
            for title in project_titles
        ],
    )


def test_shared_project_title_creates_a_symmetric_edge():
    a = make_profile("a", ["Nordic Retail Digital Transformation"])
    b = make_profile("b", ["Nordic Retail Digital Transformation"])

    graph = build_co_delivery_graph([a, b])

    assert graph["a"]["b"] == 1
    assert graph["b"]["a"] == 1


def test_edge_weight_counts_every_shared_project():
    a = make_profile("a", ["Project X", "Project Y", "Project Z"])
    b = make_profile("b", ["Project X", "Project Y", "Unrelated Project"])

    graph = build_co_delivery_graph([a, b])

    assert graph["a"]["b"] == 2
    assert graph["b"]["a"] == 2


def test_title_match_is_case_and_whitespace_insensitive():
    a = make_profile("a", ["  Global Finance Transformation  "])
    b = make_profile("b", ["global finance transformation"])

    graph = build_co_delivery_graph([a, b])

    assert graph["a"]["b"] == 1


def test_no_shared_titles_means_no_edge():
    a = make_profile("a", ["Project X"])
    b = make_profile("b", ["Project Y"])

    graph = build_co_delivery_graph([a, b])

    assert "b" not in graph.get("a", {})
    assert "a" not in graph.get("b", {})


def test_consultant_with_no_projects_has_no_edges():
    a = make_profile("a", [])
    b = make_profile("b", ["Project X"])

    graph = build_co_delivery_graph([a, b])

    assert graph.get("a", {}) == {}


def test_three_way_shared_project_links_every_pair():
    a = make_profile("a", ["Shared Project"])
    b = make_profile("b", ["Shared Project"])
    c = make_profile("c", ["Shared Project"])

    graph = build_co_delivery_graph([a, b, c])

    assert graph["a"]["b"] == 1
    assert graph["a"]["c"] == 1
    assert graph["b"]["c"] == 1
