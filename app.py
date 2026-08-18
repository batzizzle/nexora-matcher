"""Streamlit UI: turn a free-text staffing brief into an explained, ranked team recommendation.

What this does: gives a person a text box for a staffing request ("ERP change
management, manufacturing, 24 weeks from September"), a "Find team" button, and
shows the resulting recommended team as expandable, evidence-backed cards, plus two
alternative teams and a data-quality panel. Every past run this session stays
in a sidebar history so a good result is never lost by trying a new brief.

Why it exists: this is the only piece of the pipeline a non-programmer ever sees --
src/brief.py, src/match.py, and src/explain.py all already work end to end, but
only as Python function calls a developer runs from a script. This file is where
that becomes something a person can actually operate live, in a case interview.

What it takes in / produces: input is one free-text brief typed into the page (or
one of three example buttons). Output is on-screen only -- no writes anywhere.
data/processed/profiles.json, personal_data.json, availability.csv are loaded once
at startup and cached; personal_data.json is joined into consultant_id -> full_name
only for on-screen display, and is never passed into src/brief.py's or
src/match.py's LLM calls (CLAUDE.md rule 5).

Assumptions and shortcuts taken:
- Session-only history (st.session_state), not a database or a file on disk -- "no
  login, no database" was an explicit design constraint, and a live demo runs in
  one continuous browser session, so history surviving a full app restart wasn't
  judged necessary.
- Both the brief-parsing step and the matching step are wrapped in their own
  @st.cache_data-decorated function, keyed on the brief text (or the parsed
  brief's own JSON) plus today's date -- deliberately for demo stability, not just
  speed: fit scores and role assignments have been observed to vary between
  identical runs of the same brief (LLM non-determinism), so resubmitting an
  already-run brief should show the same result, not re-roll it.
- The matching step's cache function lets exceptions propagate uncached (rather
  than catching and returning an error tuple, the way brief-parsing failures are
  handled) -- src/match.py's rerank_role has no retry logic, so a transient API
  failure must remain retryable on the next click, not get permanently cached as a
  failure. Brief-parsing failures are different: a genuine two-attempt schema
  validation failure is a deterministic outcome worth caching like a success.
- The "Find team" click runs parse+match in a background thread (_run_pipeline_in_
  thread) while the main script thread polls it every 0.5s and updates an st.empty()
  placeholder -- a genuinely live-ticking elapsed-time clock, not just a stopwatch
  reading only once the whole call returns. The background thread never calls any
  st.* function itself (only the main thread may -- Streamlit widgets are tied to
  the script-run thread), it only writes to a plain dict the main thread reads.
  Progress labels are still only "parsing" vs. "matching" -- getting finer-grained
  progress from inside match() itself (filtering, retrieval, concurrent per-role
  scoring) would mean instrumenting src/match.py, which this file doesn't do. Actual
  per-step durations are captured in that same dict and rendered afterwards as a
  small bar chart (render_bar_funnel) -- on a cache hit (identical brief resubmitted)
  these read as near-zero, which is an honest reflection of what happened, not a bug.
- Team availability and the consultant co-delivery network are their own always-
  visible expanders, shown before any brief has been run -- not gated behind a
  match result. When a result does exist, the network diagram highlights the
  current recommended team's members in a different colour from everyone else.
- The co-delivery network is a hand-rolled inline-SVG circular-layout diagram
  (network_layout + render_co_delivery_network), not graphviz or networkx -- neither
  is installed, and a self-contained SVG needs no dependency at all, guaranteeing it
  renders the same way the logo does (see below). The score breakdown and both
  funnels (candidate-survival and time-per-step) use st.bar_chart instead, which is
  bundled with Streamlit -- no separate plotting library either way.
- The trust badge is st.success/warning/error with a leading symbol (check / warning
  triangle / cross) prepended for a quicker visual read, not just colour, which also
  helps a colourblind viewer or a projector with washed-out colours during a demo.
- The availability view is a colour-coded snapshot table (pandas Styler), not a
  calendar, on purpose: availability.csv only carries one status, one weekly
  free-day count, and one next-free date per consultant -- a real calendar would
  imply day-by-day granularity the underlying data doesn't have.
- The header logo is a small inline SVG, not an external image file, for the same
  reason src/match.py sets HF_HUB_OFFLINE: nothing in this app should depend on a
  network resource a live demo's wifi could drop.
- Evidence bullets on a match card visibly mark src/explain.py's EvidenceQuote.
  matched_requirement=False entries as "general skill, not a direct match", and a
  card whose evidence is entirely unmatched shows an explicit caveat above the list
  -- added after a real report: an "Azure migration" brief surfaced candidates with
  zero Azure/cloud skills (their real specialists were fully_booked, thinning the
  shortlist retrieval had to rank from), and their cards showed unrelated
  high-confidence skills as unqualified "Evidence" with nothing disclosing the gap.
- Pure display-data functions (brief_chips, funnel_rows, role_shortlist_sizes,
  split_trust_flags, graph_edges, availability_rows, network_layout, format_elapsed)
  are kept separate from every st.* rendering call specifically so they're unit-
  testable the normal way -- see tests/test_app.py. The rendering functions
  themselves are not unit-tested; they're thin, and verified by running the app.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.brief import parse_brief
from src.explain import build_match_cards_for_team, is_confirmed_trust_finding
from src.graph import build_co_delivery_graph
from src.match import match
from src.schema import ConsultantProfile, MatchCard, MatchResult, ProjectBrief, ScoreBreakdown, Team

_DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"

# tunable PoC choice: three example briefs chosen to show three different real outcomes
# observed while testing src/match.py and src/explain.py this project, not arbitrary examples.
_EXAMPLE_BRIEFS = {
    "Strong fit: ERP change management": "ERP change management, manufacturing, 24 weeks from September",
    "Trade-off: Cloud migration": "Cloud migration, Nordic retailer, 12 weeks",
    "Honest failure: Quantum Cryptographer": "Quantum Cryptographer, starting Monday",
}
_DEFAULT_BRIEF_TEXT = _EXAMPLE_BRIEFS["Strong fit: ERP change management"]

_BREAKDOWN_LABELS = {
    "skill_fit": "Skill fit",
    "seniority_fit": "Seniority fit",
    "availability": "Availability",
    "industry_experience": "Industry experience",
    "team_chemistry": "Team chemistry",
}

_TRUST_LABELS = {"verified": "Verified", "unverified_claims": "Unverified claims", "flagged": "Flagged"}
_TRUST_ICONS = {"verified": "✓", "unverified_claims": "⚠", "flagged": "✕"}  # check / warning / cross

_TEAM_LABELS = {"recommended": "Recommended", "earliest_start": "Earliest start", "lowest_cost": "Lowest cost"}

_STATUS_LABELS = {"available": "Available", "partly_booked": "Partly booked", "fully_booked": "Fully booked"}
_STATUS_ORDER = {"available": 0, "partly_booked": 1, "fully_booked": 2}
_STATUS_COLORS = {"available": "#1a7f37", "partly_booked": "#9a6700", "fully_booked": "#cf222e"}  # accessible-ish green/amber/red

# Small inline SVG, not an external image -- keeps the app fully self-contained and offline (same
# reasoning as HF_HUB_OFFLINE in src/match.py: no network dependency a live demo could trip over). A
# hub node connected to two others reads as "matching one lead to a team", echoing the co-delivery graph.
_LOGO_SVG = """<svg width="52" height="52" viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Nexora Match Maker logo">
  <line x1="26" y1="12" x2="12" y2="40" stroke="#8b949e" stroke-width="2"/>
  <line x1="26" y1="12" x2="40" y2="40" stroke="#8b949e" stroke-width="2"/>
  <line x1="12" y1="40" x2="40" y2="40" stroke="#8b949e" stroke-width="2" stroke-dasharray="3,3"/>
  <circle cx="26" cy="12" r="8" fill="#1f6feb"/>
  <circle cx="12" cy="40" r="7" fill="#2ea043"/>
  <circle cx="40" cy="40" r="7" fill="#2ea043"/>
</svg>"""


# --- data loading (cached once per process) --------------------------------


@st.cache_data
def load_profiles() -> list[ConsultantProfile]:
    """Load every consultant's PII-free profile, once per process."""
    raw = json.loads((_DATA_DIR / "profiles.json").read_text(encoding="utf-8"))
    return [ConsultantProfile.model_validate(p) for p in raw]


@st.cache_data
def load_personal_data() -> dict[str, str]:
    """Load consultant_id -> full_name only, for on-screen display -- never passed into any LLM call."""
    raw = json.loads((_DATA_DIR / "personal_data.json").read_text(encoding="utf-8"))
    return {p["consultant_id"]: p["full_name"] for p in raw}


@st.cache_data
def load_availability() -> list[dict]:
    """Load the synthetic bench-status roster, once per process."""
    with (_DATA_DIR / "availability.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["free_days_per_week"] = int(row["free_days_per_week"])
    return rows


@st.cache_data
def load_graph() -> dict[str, dict[str, int]]:
    """Build the co-delivery graph once per process -- deterministic and cheap, but no reason to redo it per click."""
    return build_co_delivery_graph(load_profiles())


# --- cached, retry-aware pipeline steps --------------------------------------


@st.cache_data(show_spinner=False)
def _cached_parse_brief(text: str, today_iso: str) -> tuple[ProjectBrief | None, dict | None]:
    """Cached so resubmitting identical brief text doesn't re-roll the LLM or change the result."""
    return parse_brief(text, today=date.fromisoformat(today_iso))


@st.cache_data(show_spinner=False)
def _cached_match(brief_json: str, today_iso: str) -> MatchResult:
    """Cached on the parsed brief's own JSON (a reliable, explicit cache key) plus today's date.

    Deliberately lets an exception propagate uncached: a transient API failure (rerank_role has no
    retry logic) must still be retryable on the next click, not permanently remembered as a failure.
    """
    brief = ProjectBrief.model_validate_json(brief_json)
    today = date.fromisoformat(today_iso)
    return match(load_profiles(), brief, load_availability(), graph=load_graph(), today=today)


def _run_pipeline_in_thread(text: str, today_iso: str, holder: dict) -> None:
    """Run parse+match in a background thread, updating `holder` only -- never calls any st.* function.

    Streamlit widgets must be called from the main script thread; this thread does no rendering, so the
    polling loop in main() (which does call st.*) can safely read `holder` while this runs, giving a
    genuinely live-ticking clock instead of a blocking call with no feedback until it returns.
    """
    timings: dict[str, float] = {}
    holder["timings"] = timings
    try:
        holder["stage"] = "parsing"
        t0 = time.perf_counter()
        brief, parse_failure = _cached_parse_brief(text, today_iso)
        timings["Parsing brief"] = time.perf_counter() - t0
        if parse_failure:
            holder["error"] = f"Couldn't parse this brief after two attempts: {parse_failure.get('error', parse_failure)}"
            holder["done"] = True
            return
        holder["brief"] = brief
        holder["stage"] = "matching"
        t1 = time.perf_counter()
        holder["result"] = _cached_match(brief.model_dump_json(), today_iso)
        timings["Matching & scoring"] = time.perf_counter() - t1
    except Exception as exc:  # rerank_role has no retry logic -- surface it, don't crash the app
        holder["error"] = (
            "Something went wrong while matching -- this can happen on a transient API issue. "
            f"Try Find team again.\n\n{exc}"
        )
    finally:
        holder["done"] = True


# --- pure display-data functions (unit-tested in tests/test_app.py) --------


def brief_chips(brief: ProjectBrief) -> list[dict]:
    """Flatten a ProjectBrief's scalar fields into (label, value, inferred) rows for chip display."""
    inferred = set(brief.inferred_fields)
    return [
        {"label": "Client", "value": brief.client, "inferred": "client" in inferred},
        {"label": "Industry", "value": brief.industry, "inferred": "industry" in inferred},
        {"label": "Start date", "value": brief.start_date or "Not specified", "inferred": "start_date" in inferred},
        {
            "label": "Duration",
            "value": f"{brief.duration_weeks} weeks" if brief.duration_weeks else "Not specified",
            "inferred": "duration_weeks" in inferred,
        },
        {"label": "Location", "value": brief.location or "Not specified", "inferred": "location" in inferred},
        {
            "label": "Required language",
            "value": brief.required_language or "Not specified",
            "inferred": "required_language" in inferred,
        },
        {
            "label": "Must-have skills",
            "value": ", ".join(brief.must_have_skills) or "None",
            "inferred": "must_have_skills" in inferred,
        },
        {"label": "Budget", "value": brief.budget or "Not disclosed", "inferred": "budget" in inferred},
    ]


def funnel_rows(result: MatchResult) -> list[tuple[str, int]]:
    """Turn MatchResult.funnel into (readable label, count) pairs, in stage order, from real data only."""
    return [(stage.stage.replace("_", " ").title(), stage.survived) for stage in result.funnel]


def role_shortlist_sizes(result: MatchResult) -> dict[str, int]:
    """How many candidates were actually scored per role (up to top-15) -- read from data already returned."""
    return {title: len(rankings) for title, rankings in result.role_rankings.items()}


def format_elapsed(seconds: float) -> str:
    """Render an elapsed-time float as a short human string, e.g. '1m 16s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


def split_trust_flags(profiles: list[ConsultantProfile]) -> tuple[list[dict], list[dict]]:
    """Split every trust_flags entry across the corpus into (confirmed findings, extraction notes).

    Uses the same distinction as src/explain.py's classify_trust (is_confirmed_trust_finding), so this
    panel never contradicts the trust badges shown elsewhere on the page.
    """
    confirmed: list[dict] = []
    notes: list[dict] = []
    for profile in profiles:
        for flag in profile.trust_flags:
            row = {"consultant_id": profile.consultant_id, "flag": flag}
            (confirmed if is_confirmed_trust_finding(flag) else notes).append(row)
    return confirmed, notes


def graph_edges(graph: dict[str, dict[str, int]]) -> list[dict]:
    """Every co-delivery edge in src/graph.py's adjacency dict, deduped (a-b and b-a collapsed to one row).

    The graph itself only exists as adjacency data for src/match.py's/src/explain.py's scoring -- nothing
    upstream ever needed a flat edge list, so this is new, app-only logic, not a reused pure function.
    """
    seen: set[frozenset[str]] = set()
    rows: list[dict] = []
    for consultant_a, neighbors in graph.items():
        for consultant_b, weight in neighbors.items():
            key = frozenset((consultant_a, consultant_b))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"a": consultant_a, "b": consultant_b, "shared_projects": weight})
    rows.sort(key=lambda row: row["shared_projects"], reverse=True)
    return rows


def availability_rows(profiles: list[ConsultantProfile], availability_by_id: dict, names_by_id: dict[str, str]) -> list[dict]:
    """One row per consultant with known availability, sorted available -> partly booked -> fully booked.

    This is a snapshot heatmap, not a real calendar: availability.csv only carries a current bench status,
    a weekly free-day count, and one next-free date per consultant -- there's no per-day future schedule in
    the data to draw an actual calendar from without fabricating granularity that isn't really there.
    """
    rows = []
    for profile in profiles:
        row = availability_by_id.get(profile.consultant_id)
        if row is None:
            continue
        rows.append(
            {
                "name": names_by_id.get(profile.consultant_id, profile.consultant_id),
                "consultant_id": profile.consultant_id,
                "status": row["status"],
                "free_days_per_week": row["free_days_per_week"],
                "next_free_date": row["next_free_date"],
            }
        )
    rows.sort(key=lambda row: (_STATUS_ORDER.get(row["status"], 99), row["next_free_date"]))
    return rows


def network_layout(graph: dict[str, dict[str, int]], names_by_id: dict[str, str], size: float = 420.0) -> dict:
    """Compute a simple circular-layout node/edge diagram for the co-delivery graph.

    Only includes consultants with at least one edge -- isolated nodes would add clutter without adding
    any "who worked with whom" information, which is the entire point of this view. Pure geometry, no
    rendering: kept separate so the layout math is unit-testable without a Streamlit runtime.
    """
    node_ids = sorted({consultant_id for consultant_id, neighbors in graph.items() if neighbors})
    count = len(node_ids)
    center = size / 2
    radius = size / 2 - 60

    positions: dict[str, tuple[float, float]] = {}
    nodes = []
    for i, node_id in enumerate(node_ids):
        angle = (2 * math.pi * i / count) if count else 0.0
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        positions[node_id] = (x, y)
        nodes.append({"id": node_id, "name": names_by_id.get(node_id, node_id), "x": x, "y": y})

    seen: set[frozenset[str]] = set()
    edges = []
    for node_id in node_ids:
        for neighbor_id, weight in graph.get(node_id, {}).items():
            key = frozenset((node_id, neighbor_id))
            if key in seen or neighbor_id not in positions:
                continue
            seen.add(key)
            x1, y1 = positions[node_id]
            x2, y2 = positions[neighbor_id]
            edges.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "weight": weight})

    return {"nodes": nodes, "edges": edges, "size": size}


# --- rendering ---------------------------------------------------------------


def render_header() -> None:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:0.25rem;">
            {_LOGO_SVG}
            <div style="font-size:2rem; font-weight:700; line-height:1.15;">
                Hello, welcome to Nexora's Match Maker
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Type a plain-English staffing brief -- role, industry, timeline -- and this tool turns it into a "
        "ranked, evidence-backed consultant team in about a minute: click **Find team** below, then open "
        "each recommendation to see its score breakdown, the CV evidence behind it, a trust badge, and who "
        "the next-best alternative would have been."
    )


def render_brief_chips(brief: ProjectBrief) -> None:
    rows = brief_chips(brief)
    cols = st.columns(4)
    for i, row in enumerate(rows):
        marker = "  *(inferred)*" if row["inferred"] else ""
        with cols[i % 4]:
            st.markdown(f"**{row['label']}**{marker}  \n{row['value']}")

    roles_marker = "  *(inferred)*" if "roles_needed" in brief.inferred_fields else ""
    st.markdown(f"**Roles needed**{roles_marker}")
    for role in brief.roles_needed:
        skills = ", ".join(role.required_skills) or "none listed"
        st.markdown(f"- {role.title} ({role.seniority}, x{role.count}) -- skills: {skills}")


def render_bar_funnel(rows: list[tuple[str, float]], value_label: str) -> None:
    """A horizontal bar chart in given order (not re-sorted) -- used for both the candidate and time funnels."""
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=["Step", value_label]).set_index("Step")
    st.bar_chart(frame, horizontal=True, sort=False)


def render_funnel(result: MatchResult) -> None:
    rows = funnel_rows(result)
    cols = st.columns(len(rows))
    for col, (label, count) in zip(cols, rows):
        col.metric(label, count)
    render_bar_funnel(rows, "Candidates")


def render_score_breakdown(breakdown: ScoreBreakdown) -> None:
    rows = [(label, getattr(breakdown, field)) for field, label in _BREAKDOWN_LABELS.items()]
    render_bar_funnel(rows, "Score")


def render_trust_badge(badge: str) -> None:
    label = f"{_TRUST_ICONS.get(badge, '')} {_TRUST_LABELS[badge]}"
    if badge == "verified":
        st.success(label)
    elif badge == "unverified_claims":
        st.warning(label)
    else:
        st.error(label)


def render_match_card(card: MatchCard, names_by_id: dict[str, str]) -> None:
    name = names_by_id.get(card.consultant_id, card.consultant_id)
    with st.expander(f"{name} -- {card.role_title} (score: {card.overall_score:.0f}/100)"):
        render_trust_badge(card.trust_badge)

        st.markdown("**Score breakdown**")
        render_score_breakdown(card.breakdown)

        st.markdown("**Evidence**")
        if card.evidence and not any(q.matched_requirement for q in card.evidence):
            st.caption(
                "⚠ None of this candidate's skills directly match what this role asked for -- showing "
                "their strongest general skills instead, so the score still carries evidence (see CLAUDE.md "
                "rule 4), but none of it should be read as proof of fit for this specific requirement."
            )
        for quote in card.evidence:
            project_note = f" _(from: {quote.project_title})_" if quote.project_title else ""
            match_note = "" if quote.matched_requirement else " _(general skill, not a direct match)_"
            st.markdown(f"- **{quote.skill_name}**{match_note}: “{quote.quote}”{project_note}")

        if card.counterfactual:
            cf = card.counterfactual
            alt_name = names_by_id.get(cf.consultant_id, cf.consultant_id)
            st.markdown(f"**Next-best alternative:** {alt_name} (fit {cf.fit_score:.0f}/100)")
            st.caption(cf.summary)

        if card.availability_alternative:
            alt = card.availability_alternative
            alt_name = names_by_id.get(alt.consultant_id, alt.consultant_id)
            st.markdown(
                f"**Worth waiting for?** {alt_name} -- free {alt.next_free_date} "
                f"({alt.days_after_start} days after your requested start)"
            )
            st.caption(alt.summary)


def render_team_gaps(team: Team) -> None:
    if not team.gaps:
        return
    lines = [f"- **{gap.role_title}** ({gap.reason.replace('_', ' ')}): {gap.detail}" for gap in team.gaps]
    st.warning("**Could not confidently staff this team:**\n" + "\n".join(lines))


def render_team(
    team: Team,
    brief: ProjectBrief,
    role_rankings: dict,
    profiles_by_id: dict[str, ConsultantProfile],
    availability_by_id: dict,
    graph: dict,
    tradeoffs: list,
    names_by_id: dict[str, str],
) -> None:
    render_team_gaps(team)
    cards = build_match_cards_for_team(team, brief, role_rankings, profiles_by_id, availability_by_id, graph, tradeoffs)
    for card in cards:
        render_match_card(card, names_by_id)


def render_data_quality(profiles: list[ConsultantProfile], names_by_id: dict[str, str]) -> None:
    confirmed, notes = split_trust_flags(profiles)

    st.markdown("**Confirmed findings** -- structured, non-benign results from src/trust.py's scan")
    if not confirmed:
        st.caption("None in this corpus.")
    for row in confirmed:
        name = names_by_id.get(row["consultant_id"], row["consultant_id"])
        st.error(f"{name} ({row['consultant_id']}): {row['flag']}")

    st.markdown("**Extraction notes** -- self-reported data-quality caveats, not trust/security concerns")
    if not notes:
        st.caption("None in this corpus.")
    for row in notes:
        name = names_by_id.get(row["consultant_id"], row["consultant_id"])
        st.caption(f"{name} ({row['consultant_id']}): {row['flag']}")


def render_co_delivery_network(graph: dict[str, dict[str, int]], names_by_id: dict[str, str], highlight_ids: set[str]) -> None:
    """A hand-rolled inline-SVG node/edge diagram -- no graphviz/networkx dependency, guaranteed to render.

    Blue nodes are the current recommended team's members (if any run has happened yet); green nodes are
    everyone else with at least one shared-project connection. Line thickness scales with shared-project count.
    """
    layout = network_layout(graph, names_by_id)
    if not layout["nodes"]:
        st.caption("No shared-project history found across the corpus.")
        return

    size = layout["size"]
    parts = [
        f'<svg width="100%" viewBox="0 0 {size:.0f} {size:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:{size:.0f}px;" role="img" '
        f'aria-label="Consultant co-delivery network">'
    ]
    for edge in layout["edges"]:
        stroke_width = 1 + edge["weight"]
        mx, my = (edge["x1"] + edge["x2"]) / 2, (edge["y1"] + edge["y2"]) / 2
        parts.append(
            f'<line x1="{edge["x1"]:.1f}" y1="{edge["y1"]:.1f}" x2="{edge["x2"]:.1f}" y2="{edge["y2"]:.1f}" '
            f'stroke="#8b949e" stroke-width="{stroke_width}" />'
        )
        parts.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="10" fill="#57606a">{edge["weight"]}</text>')
    for node in layout["nodes"]:
        color = "#1f6feb" if node["id"] in highlight_ids else "#2ea043"
        parts.append(f'<circle cx="{node["x"]:.1f}" cy="{node["y"]:.1f}" r="9" fill="{color}" />')
        parts.append(
            f'<text x="{node["x"]:.1f}" y="{node["y"] - 14:.1f}" font-size="11" text-anchor="middle" '
            f'fill="#24292f">{node["name"]}</text>'
        )
    parts.append("</svg>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    if highlight_ids:
        st.caption("Blue = current recommended team. Green = everyone else with shared-project history.")


def render_co_delivery_graph(graph: dict[str, dict[str, int]], names_by_id: dict[str, str]) -> None:
    st.caption(
        "Every pair of consultants whose project lists share a project title (src/graph.py) -- the signal "
        "the recommended team's \"team chemistry\" score rewards."
    )
    edges = graph_edges(graph)
    if not edges:
        st.caption("No shared-project history found across the corpus.")
        return
    for edge in edges:
        name_a = names_by_id.get(edge["a"], edge["a"])
        name_b = names_by_id.get(edge["b"], edge["b"])
        st.markdown(f"- **{name_a}** & **{name_b}** -- {edge['shared_projects']} shared project(s)")


def render_availability_heatmap(profiles: list[ConsultantProfile], availability_by_id: dict, names_by_id: dict[str, str]) -> None:
    rows = availability_rows(profiles, availability_by_id, names_by_id)
    if not rows:
        st.caption("No availability data loaded.")
        return

    st.caption(
        "A snapshot of current bench status, not a real calendar -- availability.csv (src/availability.py) "
        "only carries one status, one weekly free-day count, and one next-free date per consultant."
    )
    frame = pd.DataFrame(rows)[["name", "status", "free_days_per_week", "next_free_date"]]
    frame.columns = ["Consultant", "Status", "Free days / week", "Next free"]
    frame["Status"] = frame["Status"].map(_STATUS_LABELS)

    def _color_status(value: str) -> str:
        key = {v: k for k, v in _STATUS_LABELS.items()}.get(value)
        color = _STATUS_COLORS.get(key, "#8b949e")
        return f"background-color: {color}; color: white;"

    styled = frame.style.map(_color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# --- app -----------------------------------------------------------------


_STAGE_LABELS = {"parsing": "Parsing brief...", "matching": "Scoring roles against the candidate pool..."}


def main() -> None:
    st.set_page_config(page_title="Nexora Matcher", layout="wide")
    render_header()

    profiles = load_profiles()
    names_by_id = load_personal_data()
    availability_by_id = {row["consultant_id"]: row for row in load_availability()}
    graph = load_graph()
    profiles_by_id = {p.consultant_id: p for p in profiles}

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("current", None)
    st.session_state.setdefault("brief_text", _DEFAULT_BRIEF_TEXT)

    with st.sidebar:
        st.header("History")
        if not st.session_state.history:
            st.caption("Past runs will appear here as you try briefs.")
        for i, entry in enumerate(st.session_state.history):
            label = f"{entry['timestamp'].strftime('%H:%M:%S')} -- {entry['text'][:40]}"
            if st.button(label, key=f"history_{i}", use_container_width=True):
                st.session_state.current = entry

    st.subheader("📝 1. Project brief")
    example_cols = st.columns(len(_EXAMPLE_BRIEFS))
    for col, (label, text) in zip(example_cols, _EXAMPLE_BRIEFS.items()):
        if col.button(label, use_container_width=True):
            st.session_state.brief_text = text

    st.text_area("Describe the staffing need", key="brief_text", height=100)
    find_team_clicked = st.button("Find team", type="primary")

    if find_team_clicked:
        text = st.session_state.brief_text
        today = date.today()

        holder: dict = {"done": False, "stage": "parsing"}
        thread = threading.Thread(target=_run_pipeline_in_thread, args=(text, today.isoformat(), holder), daemon=True)
        start = time.perf_counter()
        thread.start()

        st.caption(
            "Matching typically takes 60-150s depending on how many roles the brief needs -- measured live "
            "against this dataset, not an estimate."
        )
        stage_placeholder = st.empty()
        clock_placeholder = st.empty()
        while not holder["done"]:
            elapsed = time.perf_counter() - start
            stage_placeholder.info(_STAGE_LABELS.get(holder["stage"], "Working..."))
            clock_placeholder.metric("Elapsed", format_elapsed(elapsed))
            time.sleep(0.5)
        thread.join()
        elapsed = time.perf_counter() - start
        clock_placeholder.metric("Elapsed", format_elapsed(elapsed))

        if holder.get("error"):
            stage_placeholder.error(holder["error"])
            st.stop()

        stage_placeholder.success("Done")
        brief, result = holder["brief"], holder["result"]
        entry = {
            "timestamp": datetime.now(), "text": text, "brief": brief, "result": result,
            "elapsed": elapsed, "timings": holder.get("timings", {}),
        }
        st.session_state.history.insert(0, entry)
        st.session_state.current = entry

    current = st.session_state.current

    highlight_ids: set[str] = set()
    if current is not None:
        recommended_team = next((t for t in current["result"].teams if t.label == "recommended"), None)
        if recommended_team is not None:
            highlight_ids = {m.consultant_id for m in recommended_team.members}

    with st.expander("📅 Team availability", expanded=False):
        render_availability_heatmap(profiles, availability_by_id, names_by_id)
    with st.expander("🔗 Consultant network (team chemistry)", expanded=False):
        render_co_delivery_network(graph, names_by_id, highlight_ids)
        render_co_delivery_graph(graph, names_by_id)

    if current is None:
        return

    brief: ProjectBrief = current["brief"]
    result: MatchResult = current["result"]
    st.caption(f"Time to result: {format_elapsed(current['elapsed'])}")
    timings = current.get("timings") or {}
    if timings:
        st.caption("Time spent per step:")
        render_bar_funnel(list(timings.items()), "Seconds")

    st.subheader("📋 2. Parsed brief")
    render_brief_chips(brief)
    render_funnel(result)

    shortlist_sizes = role_shortlist_sizes(result)
    if shortlist_sizes:
        st.caption(
            "Shortlisted for scoring: "
            + ", ".join(f"{title} ({n})" for title, n in shortlist_sizes.items())
        )

    st.subheader("👥 3. Recommended team")
    recommended = next(t for t in result.teams if t.label == "recommended")
    render_team(
        recommended, brief, result.role_rankings, profiles_by_id, availability_by_id, graph,
        result.availability_tradeoffs, names_by_id,
    )

    st.subheader("📊 4. Alternatives & data quality")
    earliest = next(t for t in result.teams if t.label == "earliest_start")
    lowest_cost = next(t for t in result.teams if t.label == "lowest_cost")

    tab_earliest, tab_cheapest, tab_quality = st.tabs(
        [_TEAM_LABELS["earliest_start"], _TEAM_LABELS["lowest_cost"], "Data quality"]
    )
    with tab_earliest:
        render_team(
            earliest, brief, result.role_rankings, profiles_by_id, availability_by_id, graph,
            result.availability_tradeoffs, names_by_id,
        )
    with tab_cheapest:
        render_team(
            lowest_cost, brief, result.role_rankings, profiles_by_id, availability_by_id, graph,
            result.availability_tradeoffs, names_by_id,
        )
    with tab_quality:
        render_data_quality(profiles, names_by_id)


if __name__ == "__main__":
    # Streamlit's script runner sets __name__ to "__main__" for the script `streamlit run` executes,
    # so this both runs the app normally and lets tests/test_app.py import the pure functions above
    # without triggering the whole app (st.set_page_config, data loads, rendering) on import.
    main()
