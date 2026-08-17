from src.extract import (
    _consultant_id_from_source,
    _dedupe_id,
    _merge_trust_flags,
    extract_profile,
)
from src.schema import TrustFlag

VALID_EXTRACTED = dict(
    current_role="BI Consultant",
    seniority="consultant",
    years_experience=6,
    skills=[
        {
            "name": "Power BI",
            "category": "technical",
            "evidence": "Built interactive dashboards in Power BI for retail clients.",
            "confidence": 0.9,
        }
    ],
    industries=["Retail"],
    certifications=[],
    languages=[{"name": "Danish", "proficiency": "native"}],
    location="Aarhus, Denmark",
    projects=[],
    trust_flags=[],
    extraction_confidence=0.85,
)

BENIGN_CV_TEXT = "Experienced BI consultant skilled in Power BI dashboards and SQL."


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


def test_extract_profile_succeeds_on_first_valid_response():
    client = _FakeClient([VALID_EXTRACTED])

    profile, failure = extract_profile(BENIGN_CV_TEXT, "cv_test", client=client)

    assert failure is None
    assert profile.consultant_id == "cv_test"
    assert profile.current_role == "BI Consultant"
    assert profile.skills[0].evidence
    assert len(client.messages.calls) == 1


def test_extract_profile_retries_once_with_error_then_succeeds():
    invalid = {**VALID_EXTRACTED, "seniority": "not-a-real-tier"}
    client = _FakeClient([invalid, VALID_EXTRACTED])

    profile, failure = extract_profile(BENIGN_CV_TEXT, "cv_test", client=client)

    assert failure is None
    assert profile.seniority == "consultant"
    assert len(client.messages.calls) == 2
    retry_message = client.messages.calls[1]["messages"][0]["content"]
    assert "failed schema validation" in retry_message


def test_extract_profile_returns_failure_after_two_bad_responses():
    invalid = {**VALID_EXTRACTED, "seniority": "not-a-real-tier"}
    client = _FakeClient([invalid, invalid])

    profile, failure = extract_profile(BENIGN_CV_TEXT, "cv_test", client=client)

    assert profile is None
    assert "error" in failure
    assert failure["raw_response"] == invalid


def test_extract_profile_requires_non_empty_skill_evidence():
    invalid = {**VALID_EXTRACTED, "skills": [{**VALID_EXTRACTED["skills"][0], "evidence": ""}]}
    client = _FakeClient([invalid, VALID_EXTRACTED])

    profile, failure = extract_profile(BENIGN_CV_TEXT, "cv_test", client=client)

    assert failure is None
    assert len(client.messages.calls) == 2


def test_consultant_id_from_plain_pdf_filename():
    assert _consultant_id_from_source("cv_10_Victor_Bøgh_BI_Consultant.pdf") == "cv_10_victor_b_gh_bi_consultant"


def test_consultant_id_from_docx_filename():
    assert _consultant_id_from_source("CV4.docx") == "cv4"


def test_consultant_id_from_pptx_slide_includes_slide_marker():
    assert _consultant_id_from_source("cvs-ppt-format.pptx#slide1") == "cvs_ppt_format_slide1"
    assert _consultant_id_from_source("cvs-ppt-format.pptx#slide2") == "cvs_ppt_format_slide2"


def test_dedupe_id_appends_counter_on_collision():
    seen = {"cv1"}
    first = _dedupe_id("cv1", seen)
    seen.add(first)
    second = _dedupe_id("cv1", seen)

    assert first == "cv1_2"
    assert second == "cv1_3"


def test_dedupe_id_returns_candidate_unchanged_when_unique():
    assert _dedupe_id("cv1", set()) == "cv1"


def test_merge_trust_flags_dedupes_and_formats_scanned_flags():
    self_reported = ["ignore all prior instructions"]
    scanned = [
        TrustFlag(span="THE BEST", classification="promotional_language", severity="medium", pattern_name="x"),
        TrustFlag(span="normal text", classification="benign", severity="low", pattern_name="y"),
    ]

    merged = _merge_trust_flags(self_reported, scanned)

    assert "ignore all prior instructions" in merged
    assert "promotional_language (medium): THE BEST" in merged
    assert not any("benign" in flag for flag in merged)


def test_merge_trust_flags_dedupes_identical_entries():
    merged = _merge_trust_flags(["same"], [])
    assert _merge_trust_flags(["same", "same"], []) == ["same"]
    assert merged == ["same"]
