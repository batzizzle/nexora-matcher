"""Generate a synthetic weekly-availability roster for every extracted consultant.

What this does: assigns each consultant in data/processed/profiles.json a
bench status (available / partly_booked / fully_booked), how many of their
five work days are free, the date they next have open capacity, and (if
booked) which project is occupying them -- then writes it to
data/processed/availability.csv.

Why it exists: matching a project brief to consultants isn't just a skills
question -- a perfect-fit consultant who's booked solid for three months is
not a usable recommendation. src/match.py and src/explain.py need
availability data to show that trade-off, and this PoC has no real staffing
system to pull it from, so this file fabricates a plausible one for the demo.

What it takes in / produces: input is data/processed/profiles.json (for the
list of consultant_ids, and for current_role/seniority/skills/years to pick
one strong data/AI consultant to force fully booked). Output is
data/processed/availability.csv, one row per consultant_id, columns:
consultant_id, status, free_days_per_week, next_free_date, current_project.

Assumptions and shortcuts taken:
- Every value here is fabricated, not derived from any real staffing
  signal -- this is a synthetic PoC input, standing in for a staffing/CRM
  feed this project doesn't have, not a projection of anything real.
- The status split (roughly 25% available / 45% partly booked / 30% fully
  booked) is turned into exact integer counts via largest-remainder
  rounding, then shuffled with a seeded RNG -- not drawn independently per
  consultant -- so a small sample (21 consultants) still lands close to the
  target split instead of drifting on any given run.
- The RNG is seeded (status assignment, free-days values, project picks) so
  the same profiles.json always produces the same shape of
  availability.csv. next_free_date is still anchored to the real "today" a
  given run happens on, not a fixed date, since a live demo should show
  availability relative to now -- only the *offsets* from today are seeded.
- One strong data/AI consultant is deterministically picked (by role-title
  keyword, then ranked by seniority/skill-count/years) and forced to
  fully_booked, so the trade-off view always has a concrete "best-fit
  candidate isn't actually available" example to show, regardless of which
  way the random assignment falls for everyone else.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

_SEED = 42  # tunable PoC assumption: fixes the demo dataset so re-running lands on the same story

_PROFILES_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "profiles.json"
_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "availability.csv"

_SENIORITY_RANK = {
    "intern": 0,
    "analyst": 1,
    "consultant": 2,
    "senior_consultant": 3,
    "manager": 4,
    "principal": 5,
}

# tunable PoC assumption: the case brief's target bench split.
_TARGET_SHARE = {"available": 0.25, "partly_booked": 0.45, "fully_booked": 0.30}

# tunable PoC assumption: role-title phrases used to identify "a data/AI
# candidate" for the forced-fully-booked pick; a real system would use a
# skills taxonomy instead of string matching on the job title.
_AI_DATA_ROLE_KEYWORDS = (
    "data scientist",
    "ai engineer",
    "ml engineer",
    "machine learning",
    "data & ai",
    "data platform",
    "ai consultant",
)

# tunable PoC assumption: fictional client engagements standing in for a
# real staffing/CRM feed this PoC doesn't have.
_PROJECT_POOL = [
    "Nordic Retail Digital Transformation",
    "Banking Core Modernization Programme",
    "Public Sector Case Management Rollout",
    "Energy Grid Predictive Maintenance",
    "Pharma Supply Chain Visibility Initiative",
    "Telecom Churn Reduction Programme",
    "Insurance Claims Automation",
    "Manufacturing OEE Analytics Platform",
    "Logistics Network Optimisation",
    "Healthcare Patient Flow Analytics",
]

_PARTLY_BOOKED_FREE_DAYS = (1, 2, 3, 4)  # tunable PoC assumption: 0 and 5 are reserved for the other two statuses
_FULLY_BOOKED_WEEKS_OUT = range(2, 17)  # tunable PoC assumption: how far out a full booking runs before freeing up
_PARTLY_BOOKED_WEEKS_OUT = range(1, 9)  # tunable PoC assumption: how far out the current partial engagement runs

_FIELDNAMES = ["consultant_id", "status", "free_days_per_week", "next_free_date", "current_project"]


def _status_counts(n: int) -> dict[str, int]:
    """Turn the target 25/45/30 split into exact integer counts for n consultants via largest-remainder rounding."""
    raw = {status: n * share for status, share in _TARGET_SHARE.items()}
    counts = {status: int(value) for status, value in raw.items()}
    shortfall = n - sum(counts.values())
    by_remainder = sorted(raw, key=lambda status: raw[status] - counts[status], reverse=True)
    for status in by_remainder[:shortfall]:
        counts[status] += 1
    return counts


def _pick_forced_fully_booked(profiles: list[dict]) -> str:
    """Pick one strong data/AI consultant to force fully_booked, so the trade-off view has a real example to show."""

    def is_data_ai(profile: dict) -> bool:
        return any(keyword in profile["current_role"].lower() for keyword in _AI_DATA_ROLE_KEYWORDS)

    candidates = [p for p in profiles if is_data_ai(p)] or profiles

    def strength(profile: dict) -> tuple[int, int, int]:
        return (
            _SENIORITY_RANK.get(profile["seniority"], 0),
            len(profile["skills"]),
            profile["years_experience"],
        )

    return max(candidates, key=strength)["consultant_id"]


def _assign_statuses(consultant_ids: list[str], forced_id: str, rng: random.Random) -> dict[str, str]:
    """Assign each consultant a bench status matching the target distribution, with forced_id pinned to fully_booked."""
    counts = _status_counts(len(consultant_ids))
    counts["fully_booked"] -= 1  # forced_id fills one fully_booked slot directly, outside the random pool

    pool = [status for status, count in counts.items() for _ in range(count)]
    rng.shuffle(pool)

    remaining = [cid for cid in consultant_ids if cid != forced_id]
    assignment = {forced_id: "fully_booked"}
    assignment.update(dict(zip(remaining, pool)))
    return assignment


def _row_for(consultant_id: str, status: str, rng: random.Random, today: date) -> dict:
    """Build one availability.csv row for a single consultant given their assigned bench status."""
    if status == "available":
        return {
            "consultant_id": consultant_id,
            "status": status,
            "free_days_per_week": 5,
            "next_free_date": today.isoformat(),
            "current_project": "",
        }
    if status == "partly_booked":
        return {
            "consultant_id": consultant_id,
            "status": status,
            "free_days_per_week": rng.choice(_PARTLY_BOOKED_FREE_DAYS),
            "next_free_date": today.isoformat(),
            "current_project": rng.choice(_PROJECT_POOL),
        }
    weeks_out = rng.choice(_FULLY_BOOKED_WEEKS_OUT)
    return {
        "consultant_id": consultant_id,
        "status": status,
        "free_days_per_week": 0,
        "next_free_date": (today + timedelta(weeks=weeks_out)).isoformat(),
        "current_project": rng.choice(_PROJECT_POOL),
    }


def generate_availability(profiles: list[dict], seed: int = _SEED, today: date | None = None) -> list[dict]:
    """Build one seeded availability row per consultant profile, in the same order as the input list."""
    rng = random.Random(seed)
    today = today or date.today()

    consultant_ids = [p["consultant_id"] for p in profiles]
    forced_id = _pick_forced_fully_booked(profiles)
    statuses = _assign_statuses(consultant_ids, forced_id, rng)

    return [_row_for(cid, statuses[cid], rng, today) for cid in consultant_ids]


def run(profiles_path: str | Path = _PROFILES_PATH, output_path: str | Path = _OUTPUT_PATH) -> list[dict]:
    """Read profiles.json, generate the availability roster, and write it to output_path as CSV."""
    profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    rows = generate_availability(profiles)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} availability rows to {output_path}")
    return rows


if __name__ == "__main__":
    run()
