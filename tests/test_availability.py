import csv
import json
from datetime import date
from pathlib import Path

import pytest

from src.availability import (
    _FIELDNAMES,
    _assign_statuses,
    _pick_forced_fully_booked,
    _status_counts,
    generate_availability,
    run,
)

PROFILES_PATH = Path(__file__).parent.parent / "data" / "processed" / "profiles.json"

pytestmark = pytest.mark.skipif(
    not PROFILES_PATH.exists(),
    reason="requires data/processed/profiles.json from a prior `python -m src.extract` run",
)


@pytest.fixture(scope="module")
def profiles():
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def make_profile(**overrides):
    defaults = dict(
        consultant_id="cv_x",
        current_role="Management Consultant",
        seniority="consultant",
        years_experience=5,
        skills=[],
        location="Copenhagen, Denmark",
        extraction_confidence=0.9,
    )
    return {**defaults, **overrides}


def test_status_counts_matches_target_split_for_21_consultants():
    counts = _status_counts(21)
    assert counts == {"available": 5, "partly_booked": 10, "fully_booked": 6}
    assert sum(counts.values()) == 21


def test_status_counts_sums_to_n_for_arbitrary_sizes():
    for n in (1, 2, 7, 13, 40):
        assert sum(_status_counts(n).values()) == n


def test_pick_forced_fully_booked_prefers_data_ai_role_keyword():
    profiles = [
        make_profile(consultant_id="a", current_role="Change Management Consultant"),
        make_profile(consultant_id="b", current_role="Senior Data Scientist", seniority="senior_consultant"),
    ]
    assert _pick_forced_fully_booked(profiles) == "b"


def test_pick_forced_fully_booked_ranks_by_seniority_then_skills_then_years():
    junior = make_profile(
        consultant_id="junior",
        current_role="ML Engineer",
        seniority="consultant",
        years_experience=2,
        skills=[{"name": "x"}],
    )
    senior = make_profile(
        consultant_id="senior",
        current_role="AI Engineer",
        seniority="senior_consultant",
        years_experience=8,
        skills=[{"name": "x"}, {"name": "y"}],
    )
    assert _pick_forced_fully_booked([junior, senior]) == "senior"


def test_pick_forced_fully_booked_falls_back_to_all_profiles_when_no_data_ai_role():
    profiles = [make_profile(consultant_id="only", current_role="Supply Chain Consultant")]
    assert _pick_forced_fully_booked(profiles) == "only"


def test_assign_statuses_pins_forced_id_to_fully_booked_and_covers_everyone():
    import random

    ids = [f"cv_{i}" for i in range(21)]
    assignment = _assign_statuses(ids, "cv_5", random.Random(1))

    assert assignment["cv_5"] == "fully_booked"
    assert set(assignment) == set(ids)
    assert all(status in {"available", "partly_booked", "fully_booked"} for status in assignment.values())


def test_generate_availability_is_deterministic_for_a_fixed_seed_and_date(profiles):
    today = date(2026, 1, 5)
    first = generate_availability(profiles, seed=42, today=today)
    second = generate_availability(profiles, seed=42, today=today)
    assert first == second


def test_generate_availability_covers_every_consultant_exactly_once(profiles):
    rows = generate_availability(profiles)
    ids = [r["consultant_id"] for r in rows]
    assert sorted(ids) == sorted(p["consultant_id"] for p in profiles)
    assert len(ids) == len(set(ids))


def test_generate_availability_status_field_values_are_internally_consistent(profiles):
    rows = generate_availability(profiles)
    for row in rows:
        assert row["status"] in {"available", "partly_booked", "fully_booked"}
        if row["status"] == "available":
            assert row["free_days_per_week"] == 5
            assert row["current_project"] == ""
        elif row["status"] == "partly_booked":
            assert 1 <= row["free_days_per_week"] <= 4
            assert row["current_project"]
        else:
            assert row["free_days_per_week"] == 0
            assert row["current_project"]


def test_generate_availability_forces_a_strong_data_ai_candidate_fully_booked(profiles):
    rows = generate_availability(profiles)
    by_id = {r["consultant_id"]: r for r in rows}
    forced_id = _pick_forced_fully_booked(profiles)

    assert by_id[forced_id]["status"] == "fully_booked"
    assert by_id[forced_id]["free_days_per_week"] == 0
    assert by_id[forced_id]["current_project"]


def test_generate_availability_next_free_date_is_future_for_fully_booked_only(profiles):
    today = date(2026, 1, 5)
    rows = generate_availability(profiles, today=today)
    for row in rows:
        next_free = date.fromisoformat(row["next_free_date"])
        if row["status"] == "fully_booked":
            assert next_free > today
        else:
            assert next_free == today


def test_run_writes_csv_with_expected_header_and_row_count(profiles, tmp_path):
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(json.dumps(profiles), encoding="utf-8")
    output_path = tmp_path / "availability.csv"

    rows = run(profiles_path=profiles_path, output_path=output_path)

    with output_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == _FIELDNAMES
        written_rows = list(reader)

    assert len(written_rows) == len(rows) == len(profiles)
