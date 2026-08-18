from datetime import date

from src.brief import _build_user_message, _system_prompt, parse_brief

VALID_BRIEF = dict(
    client="Nordic Manufacturing Group",
    industry="Manufacturing",
    roles_needed=[
        {
            "title": "ERP Change Management Lead",
            "seniority": "senior_consultant",
            "count": 1,
            "required_skills": ["Change Management", "SAP S/4HANA"],
        }
    ],
    must_have_skills=[],
    nice_to_have_skills=[],
    start_date="September 2026",
    duration_weeks=24,
    seniority_mix={},
    location=None,
    required_language=None,
    budget=None,
    inferred_fields=["roles_needed", "client"],
)

BRIEF_TEXT = "ERP change management, manufacturing, 24 weeks from September"


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


def test_parse_brief_succeeds_on_first_valid_response():
    client = _FakeClient([VALID_BRIEF])

    brief, failure = parse_brief(BRIEF_TEXT, client=client, today=date(2026, 8, 18))

    assert failure is None
    assert brief.client == "Nordic Manufacturing Group"
    assert brief.roles_needed[0].title == "ERP Change Management Lead"
    assert "roles_needed" in brief.inferred_fields
    assert len(client.messages.calls) == 1


def test_parse_brief_retries_once_with_error_then_succeeds():
    invalid = {**VALID_BRIEF, "roles_needed": [{**VALID_BRIEF["roles_needed"][0], "seniority": "not-a-real-tier"}]}
    client = _FakeClient([invalid, VALID_BRIEF])

    brief, failure = parse_brief(BRIEF_TEXT, client=client, today=date(2026, 8, 18))

    assert failure is None
    assert brief.roles_needed[0].seniority == "senior_consultant"
    assert len(client.messages.calls) == 2
    retry_message = client.messages.calls[1]["messages"][0]["content"]
    assert "failed schema validation" in retry_message


def test_parse_brief_returns_failure_after_two_bad_responses():
    invalid = {**VALID_BRIEF, "roles_needed": [{**VALID_BRIEF["roles_needed"][0], "seniority": "not-a-real-tier"}]}
    client = _FakeClient([invalid, invalid])

    brief, failure = parse_brief(BRIEF_TEXT, client=client, today=date(2026, 8, 18))

    assert brief is None
    assert "error" in failure
    assert failure["raw_response"] == invalid


def test_parse_brief_rejects_extra_fields():
    invalid = {**VALID_BRIEF, "unexpected_field": "surprise"}
    client = _FakeClient([invalid, VALID_BRIEF])

    brief, failure = parse_brief(BRIEF_TEXT, client=client, today=date(2026, 8, 18))

    assert failure is None
    assert len(client.messages.calls) == 2


def test_build_user_message_includes_free_text_and_no_retry_note_by_default():
    message = _build_user_message(BRIEF_TEXT, retry_error=None)
    assert BRIEF_TEXT in message
    assert "failed schema validation" not in message


def test_build_user_message_appends_retry_error():
    message = _build_user_message(BRIEF_TEXT, retry_error="seniority: invalid literal")
    assert "failed schema validation" in message
    assert "seniority: invalid literal" in message


def test_system_prompt_anchors_relative_dates_to_todays_date():
    prompt = _system_prompt(date(2026, 8, 18))
    assert "2026-08-18" in prompt


def test_system_prompt_teaches_must_have_vs_required_skills_distinction():
    prompt = _system_prompt(date(2026, 8, 18))
    assert "HARD FILTER" in prompt
    assert "required_skills" in prompt
