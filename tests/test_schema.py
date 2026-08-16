import pytest
from pydantic import ValidationError

from src.schema import (
    ConsultantProfile,
    Language,
    PersonalData,
    Project,
    ProjectBrief,
    RoleRequirement,
    Skill,
)


def make_skill(**overrides):
    defaults = dict(
        name="Power BI",
        category="technical",
        evidence="Developed interactive HR analytics dashboards in Power BI.",
        confidence=0.9,
    )
    return Skill(**{**defaults, **overrides})


def make_project(**overrides):
    defaults = dict(
        title="Retail Sales Analytics Platform",
        industry="Retail",
        role="Lead Developer",
        impact="30% faster insights",
        tech=["Power BI", "Azure SQL", "DAX"],
        year_start=2020,
        year_end=2022,
    )
    return Project(**{**defaults, **overrides})


def make_consultant_profile(**overrides):
    defaults = dict(
        consultant_id="cv_10",
        current_role="BI Consultant",
        seniority="consultant",
        years_experience=5,
        skills=[make_skill()],
        industries=["Retail", "Financial Services"],
        certifications=[],
        languages=[Language(name="Danish", proficiency="native")],
        location="Aarhus, Denmark",
        projects=[make_project()],
        trust_flags=[],
        extraction_confidence=0.87,
    )
    return ConsultantProfile(**{**defaults, **overrides})


def test_skill_requires_non_empty_evidence():
    with pytest.raises(ValidationError):
        make_skill(evidence="")


def test_skill_confidence_must_be_within_unit_interval():
    with pytest.raises(ValidationError):
        make_skill(confidence=1.5)
    with pytest.raises(ValidationError):
        make_skill(confidence=-0.1)


def test_skill_rejects_invalid_category():
    with pytest.raises(ValidationError):
        make_skill(category="not-a-real-category")


def test_skill_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        make_skill(unexpected_field="oops")


def test_project_allows_ongoing_project_with_no_end_year():
    project = make_project(year_end=None)
    assert project.year_end is None


def test_consultant_profile_round_trip():
    profile = make_consultant_profile()
    payload = profile.model_dump()
    rebuilt = ConsultantProfile.model_validate(payload)
    assert rebuilt == profile


def test_consultant_profile_rejects_invalid_seniority():
    with pytest.raises(ValidationError):
        make_consultant_profile(seniority="ceo")


def test_consultant_profile_rejects_negative_years_experience():
    with pytest.raises(ValidationError):
        make_consultant_profile(years_experience=-1)


def test_consultant_profile_extraction_confidence_bounds():
    with pytest.raises(ValidationError):
        make_consultant_profile(extraction_confidence=1.01)


def test_personal_data_allows_missing_optional_contact_fields():
    personal = PersonalData(consultant_id="cv_10", full_name="Victor Bøgh")
    assert personal.email is None
    assert personal.github is None


def test_role_requirement_requires_at_least_one_headcount():
    with pytest.raises(ValidationError):
        RoleRequirement(title="Data Engineer", seniority="consultant", count=0)


def test_project_brief_seniority_mix_keys_are_validated():
    with pytest.raises(ValidationError):
        ProjectBrief(
            client="Nexora",
            industry="Banking",
            seniority_mix={"ceo": 1},
        )


def test_project_brief_minimal_construction_uses_defaults():
    brief = ProjectBrief(client="Nexora", industry="Banking")
    assert brief.roles_needed == []
    assert brief.must_have_skills == []
    assert brief.inferred_fields == []
    assert brief.start_date is None


def test_project_brief_tracks_inferred_fields():
    brief = ProjectBrief(
        client="Nexora",
        industry="Banking",
        start_date="ASAP",
        inferred_fields=["start_date"],
    )
    assert brief.inferred_fields == ["start_date"]
