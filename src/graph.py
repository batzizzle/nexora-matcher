"""Compute a co-delivery graph: which consultants have worked together before.

What this does: looks at every consultant's past-project list and draws a
weighted edge between two consultants whenever their CVs describe what looks
like the same past engagement -- an exact match on project title. The edge
weight is how many such shared projects the pair has.

Why it exists: src/match.py's team-assembly step (stage 4) rewards picking
people who have already delivered together, since a team that has worked
together before ships faster than strangers with an identical skill set.
This file is where that "have they worked together" signal is computed,
once, from data every consultant's profile already carries.

What it takes in / produces: input is the list of ConsultantProfile records
produced by src/extract.py. Output is an adjacency dict --
{consultant_id: {other_consultant_id: shared_project_count}} -- covering both
directions of every edge that has at least one shared project.

Assumptions and shortcuts taken:
- Two consultants are only linked by an exact (case-insensitive, whitespace-
  trimmed) match on Project.title. Phase 3's brief also asked for a "same
  client+year" link, but ConsultantProfile.projects has no client field --
  src/schema.py's Project model only has title/industry/role/impact/tech/
  year_start/year_end -- and the next-best proxy available, same industry +
  same year_start, is far too noisy to stand in for it: on this project's
  own 21-CV dataset it groups 20 unrelated engagements as "shared" purely
  because they happened in the same industry the same year, which would
  fabricate co-delivery bonuses between consultants who never worked
  together. Exact-title matching is sparser but not fabricated -- see
  DECISIONS.md phase 3.
- CVs describing the same real engagement don't always use identical
  wording (verified against this dataset: e.g. one consultant's "Global
  Finance Transformation and Shared Services" vs another's "...Rollout" for
  what reads like the same programme) -- so this graph under-counts true
  co-delivery rather than over-counting it. A production version would need
  fuzzy matching or a shared engagement ID from a real staffing system,
  neither of which this PoC has.
"""

from __future__ import annotations

from collections import defaultdict

from src.schema import ConsultantProfile


def _project_titles(profile: ConsultantProfile) -> set[str]:
    """Return one consultant's project titles, normalised for exact-match comparison."""
    return {p.title.strip().lower() for p in profile.projects if p.title.strip()}


def build_co_delivery_graph(profiles: list[ConsultantProfile]) -> dict[str, dict[str, int]]:
    """Build a co-delivery adjacency dict: consultant_id -> {other_consultant_id: shared_project_count}."""
    titles_by_consultant = {p.consultant_id: _project_titles(p) for p in profiles}
    graph: dict[str, dict[str, int]] = defaultdict(dict)

    ids = list(titles_by_consultant)
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1 :]:
            shared = len(titles_by_consultant[id_a] & titles_by_consultant[id_b])
            if shared:
                graph[id_a][id_b] = shared
                graph[id_b][id_a] = shared

    return dict(graph)
