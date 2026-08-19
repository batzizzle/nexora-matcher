# Nexora Consultant-to-Project Matcher (PoC)

A demo tool built for a case interview. It ingests heterogeneous consultant
CVs (PDF / DOCX / PPTX), accepts a free-text project brief, and returns a
ranked, explained team recommendation with evidence, availability trade-offs,
and alternative teams optimised for earliest start / lowest cost.

Optimised for demoability and clear reasoning, not production completeness —
see [DECISIONS.md](DECISIONS.md) for the full build history, design choices,
and known limitations, and [CLAUDE.md](CLAUDE.md) for the durable architecture
rules and stack this repo follows.

## How it works

1. **Ingest** (`src/ingest.py`) — routes a CV file (pdf/docx/pptx) to the
   right parser and extracts raw text + metadata.
2. **Trust & PII separation** (`src/trust.py`) — scans CV text for prompt-
   injection attempts and splits personal identifiers (name, email, phone,
   address, LinkedIn, GitHub) out into their own record, keyed by
   `consultant_id`.
3. **Extraction** (`src/extract.py`) — one LLM call per CV turns raw text
   into a structured, evidence-backed `ConsultantProfile` (skills, projects,
   languages, seniority, etc.), with every claim tied back to the CV
   sentence that supports it.
4. **Availability** (`src/availability.py`) — fabricates a synthetic
   weekly-availability roster per consultant (this PoC has no real staffing
   system to pull one from).
5. **Brief intake** (`src/brief.py`) — one LLM call turns a free-text project
   description into a structured `ProjectBrief` (roles needed, must-have
   skills, location, start date, budget, etc.).
6. **Matching** (`src/match.py`) — a four-stage pipeline:
   - deterministic hard filter (availability, language, must-have skills,
     location),
   - hybrid BM25 + semantic retrieval to shortlist candidates per role,
   - one LLM call per role to score and explain the shortlist,
   - deterministic greedy team assembly (rewarding complementary skills and
     past co-delivery, penalising thin availability) into a recommended team
     plus two alternatives.
7. **Explanation** (`src/explain.py`) — score breakdowns, cited evidence, and
   counterfactuals ("why not candidate X") for the UI.
8. **UI** (`app.py`) — a Streamlit app tying all of the above together.

The **LLM extracts, re-ranks, and explains — it never selects the final
team.** Hard filtering and team assembly are always plain, auditable Python.
See CLAUDE.md's non-negotiable architecture rules for the full list (JSON-
validated LLM output only, CV text always treated as untrusted data, every
score carries evidence, no PII sent to the LLM during matching).

## Repo layout

```
src/ingest.py      format router: pdf/docx/pptx -> raw text + metadata
src/trust.py        injection detection, PII extraction and separation
src/extract.py      LLM -> ConsultantProfile schema
src/availability.py synthetic bench-status roster generator
src/graph.py         co-delivery graph (who has worked together before)
src/brief.py         LLM -> ProjectBrief schema
src/match.py         filter -> retrieve -> rerank -> assemble
src/explain.py       score breakdown, evidence, counterfactual
src/schema.py         Pydantic models shared across the pipeline
app.py                Streamlit UI
data/raw/             input CVs (gitignored -- not checked in)
data/processed/       extraction output: profiles, PII, availability (gitignored, reproducible)
tests/                one pytest file per src module
```

## Setup

Requires Python 3.11+ and an Anthropic API key.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install anthropic streamlit sentence-transformers rank-bm25 \
            scikit-learn pydantic pdfplumber python-docx python-pptx \
            networkx pandas numpy pytest
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=sk-...
```

### Build the dataset

`data/raw/` and `data/processed/` are gitignored (raw CVs are personal data;
processed output is PII-shaped and fully reproducible from raw). To build the
dataset from scratch:

1. Drop CV files (`.pdf`, `.docx`, `.pptx`) into `data/raw/`.
2. Run extraction — parses, trust-scans, and extracts every CV into
   `data/processed/profiles.json` and `data/processed/personal_data.json`:
   ```bash
   python -m src.extract
   ```
3. Generate a synthetic availability roster:
   ```bash
   python -m src.availability
   ```

The co-delivery graph (`src/graph.py`) needs no separate build step — it's
computed live from `profiles.json` each time the app starts.

### Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Type (or pick an example) a project brief
and click "Find team" — a full match typically takes 60-150s depending on how
many roles the brief needs (one LLM call per role, run concurrently).

## Testing

```bash
pytest
```

Every `src/` module has a matching test file in `tests/`; tests were written
before their implementation per this project's convention.

## Known limitations

This is a PoC: synthetic availability data, no real rate cards (seniority
tier stands in as a cost proxy), single-machine SQLite/JSON storage, no
deployment story. See DECISIONS.md's "Known limitations" sections (one per
build phase) for the full, current list and what would need to change for
production.
