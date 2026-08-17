from pathlib import Path

import pytest

from src.ingest import ingest_directory
from src.schema import ConsultantProfile, Skill
from src.trust import scan_for_injection, separate_pii

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def records():
    return ingest_directory(DATA_DIR)


@pytest.fixture(scope="module")
def flags_by_source(records):
    return {r["source_file"]: scan_for_injection(r["raw_text"]) for r in records}


def test_cv4_raises_high_severity_injection_flag(flags_by_source):
    cv4_flags = flags_by_source["CV4.docx"]
    assert any(f.severity == "high" and f.classification == "injection" for f in cv4_flags)


def test_false_positive_rate_across_other_cvs(flags_by_source):
    other_files = [name for name in flags_by_source if name != "CV4.docx"]
    assert len(other_files) == 20

    clean_count = sum(
        1
        for name in other_files
        if not any(f.severity == "high" for f in flags_by_source[name])
    )
    assert clean_count >= 18


def test_flags_carry_span_severity_and_classification(flags_by_source):
    for flag in flags_by_source["CV4.docx"]:
        assert flag.span
        assert flag.severity in {"low", "medium", "high"}
        assert flag.classification in {"injection", "promotional_language", "benign"}


def test_scan_returns_empty_list_when_nothing_flagged():
    assert scan_for_injection("Experienced consultant skilled in Power BI and SQL.") == []


def make_profile(**overrides) -> ConsultantProfile:
    defaults = dict(
        consultant_id="cv_10",
        current_role="BI Consultant, victor.bogh@nexora.com, +45 32 18 27 05",
        seniority="consultant",
        years_experience=7,
        skills=[
            Skill(
                name="Power BI",
                category="technical",
                evidence="Victor Bøgh built dashboards; contact victor.bogh@nexora.com.",
                confidence=0.9,
            )
        ],
        location="Aarhus, Denmark",
        extraction_confidence=0.9,
    )
    return ConsultantProfile(**{**defaults, **overrides})


def test_separate_pii_leaves_no_email_or_danish_phone_in_profile(records):
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    raw_text = by_source["cv_10_Victor_Bøgh_BI_Consultant.pdf"]
    profile = make_profile()

    personal_data, cleaned_profile = separate_pii(raw_text, profile)

    dumped = cleaned_profile.model_dump_json()
    assert "@" not in dumped
    assert "+45" not in dumped
    assert personal_data.email == "victor.bogh@nexora.com"
    assert personal_data.full_name == "Victor Bøgh"


def test_separate_pii_returns_consultant_id_linked_pair(records):
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    raw_text = by_source["cv_10_Victor_Bøgh_BI_Consultant.pdf"]
    profile = make_profile()

    personal_data, cleaned_profile = separate_pii(raw_text, profile)

    assert personal_data.consultant_id == profile.consultant_id
    assert cleaned_profile.consultant_id == profile.consultant_id


def test_separate_pii_skips_document_header_lines_for_name(records):
    # "Curriculum Vitae" is itself two Title-Case words, so a naive name
    # heuristic picks it up instead of the real name on the very next line.
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    raw_text = by_source["Cv1.docx"]
    profile = make_profile(consultant_id="cv_alex_morgan")

    personal_data, _ = separate_pii(raw_text, profile)

    assert personal_data.full_name == "Alex Morgan"


def test_separate_pii_strips_whitespace_from_extracted_phone(records):
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    raw_text = by_source["Cv1.docx"]
    profile = make_profile(consultant_id="cv_alex_morgan")

    personal_data, _ = separate_pii(raw_text, profile)

    assert personal_data.phone == "+45 12345678"


def test_separate_pii_prefers_name_label_over_unrelated_section_heading(records):
    # CV3.docx puts the name under "Personal Information" as
    # "Name: Mikkel T. Rasmussen" -- the mid-initial period means it can't
    # match the bare Title-Case-line heuristic, which used to fall through
    # to the first unrelated Title-Case section heading further down the
    # document ("Deployment History") instead.
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    raw_text = by_source["CV3.docx"]
    profile = make_profile(consultant_id="cv3")

    personal_data, _ = separate_pii(raw_text, profile)

    assert personal_data.full_name == "Mikkel T. Rasmussen"


def test_separate_pii_prefers_name_label_when_preceded_by_other_title_case_lines(records):
    # CV5.docx has "Example Profile: Alexander Jensen" (not a name label)
    # above the real "Name: Alexander Jensen" line.
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    raw_text = by_source["CV5.docx"]
    profile = make_profile(consultant_id="cv5")

    personal_data, _ = separate_pii(raw_text, profile)

    assert personal_data.full_name == "Alexander Jensen"
