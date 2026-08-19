# Decisions

Narrative, history, and reasoning for each build phase. Durable rules, schema,
and stack facts live in CLAUDE.md instead — this file is the "why," not the
"what's currently true."

## Phase 1 — Ingestion (`src/ingest.py`)

**What was built:** a format router that reads every PDF/DOCX/PPTX in
`data/raw/` and returns one text record per CV (one per slide for PPTX decks
with multiple CVs), with source file, format, extracted text, and a
timestamp.

**Key design decisions and the alternative rejected:**
- **Patch pdfminer's glyph resolution in-process rather than post-processing
  the extracted text with a general spell-checker or transliteration pass.**
  The corruption is a specific, verifiable font-encoding bug (Danish letters
  and typographic ligatures render as either the wrong character or a bare
  `(cid:N)` placeholder), not a general OCR-quality problem. A narrow,
  evidence-backed patch fixes it exactly; a general "fix weird characters"
  pass would have been guesswork risking new silent corruption of the same
  kind it was meant to prevent.
- **Only substitute the six specific `(cid:N)` codes and Danish letter codes
  confirmed by cross-referencing surrounding text across multiple files** —
  not a blanket "fall back to WinAnsiEncoding for anything unresolved." A
  blanket fallback was tested and found to mis-decode bullet/icon glyphs
  into garbage (e.g. a bullet point rendered as a stray `ˆ`). Restricting
  the fix to codes with confirmed evidence trades completeness for
  correctness.
- **Extract headers and footers alongside the document body**, not just
  `Document.paragraphs`. Discovered only because one CV in the dataset hid a
  prompt-injection payload in white-on-white header text — invisible to a
  human and to naive paragraph-only extraction. Headers/footers are a small
  amount of extra code for a real trust-boundary gap.
- **Skip and log a file that fails to parse, rather than aborting the whole
  batch.** One malformed CV shouldn't block ingestion of the other 20.

**Assumptions made:**
- The specific glyph-encoding fixes are verified against this dataset's 14
  PDFs, not a general PDF-encoding solution — a new batch of CVs from a
  different source tool could reintroduce unresolved `(cid:N)` codes that
  need their own evidence-based mapping.
- Header/footer/first-page/even-page variants are all extracted defensively,
  even though only the plain `header` was populated in this dataset, because
  the injection-hiding technique that motivated the fix could equally target
  any of them.

**Known limitations / what would change for production:**
- No OCR fallback for scanned/image-only PDFs — `pdfplumber` returns empty
  text for those and the CV would silently produce near-empty output (still
  caught by the ≥200-char test, but not recoverable).
- The Danish-letter and ligature fixes are dataset-specific patches on a
  third-party library's internals (`pdfminer.encodingdb`); a production
  system ingesting CVs from arbitrary sources would need a more general
  font-encoding recovery strategy, or would need to detect and reject PDFs
  it can't confidently decode rather than silently substituting.

## Phase 2 (3a+3b) — Schema & trust layer (`src/schema.py`, `src/trust.py`)

**What was built:** the Pydantic schemas every LLM call and downstream
module validates against (`ConsultantProfile`, `PersonalData`,
`ProjectBrief`, `Skill`, `Project`, `Language`, `RoleRequirement`,
`TrustFlag`), plus `src/trust.py`'s two functions: `scan_for_injection`
(two-stage prompt-injection / promotional-language detection) and
`separate_pii` (PII extraction and redaction).

**Key design decisions and the alternative rejected:**
- **Two-stage injection detection: cheap regex candidate-generation, then a
  single LLM call only on files that tripped a pattern** — rejected running
  an LLM classifier over every CV unconditionally (slower and costlier for
  no benefit, since the vast majority of CVs contain nothing suspicious) and
  rejected regex-only detection (too brittle and prone to both false
  positives and false negatives for natural-language manipulation attempts).
  Verified empirically: only 1 of 21 CVs in the dataset trips any stage-1
  pattern, so the two-stage design costs exactly one real API call across
  the whole test suite.
- **Stage-1 patterns are deliberately broad (recall over precision)** —
  a false-positive candidate just gets classified "benign" by stage 2 at
  low severity; a missed real injection gets no second look at all. This
  shifts the cost of being wrong from "silent miss" to "one extra
  classification," which is the safer failure mode.
- **`extra="forbid"` on every Pydantic model** — rejected the more lenient
  default of silently ignoring unrecognized fields. An LLM hallucinating an
  extra field is a signal worth surfacing as a hard validation error, not
  data worth quietly discarding.
- **`separate_pii` redacts via regex applied directly to the profile's own
  text, not just by replacing the literal strings extracted from
  `raw_text`** — rejected relying solely on string-replace of the specific
  values pulled from the CV, because the profile's LLM-generated text
  fields (e.g. `Skill.evidence`) could in principle contain a *different*
  formatting of the same email/phone than what regex-matched in the raw
  text. Re-applying the regexes to the profile itself is what actually
  guarantees no `@` or phone-number pattern survives, rather than merely
  making it likely.
- **Ambiguous seniority-tier mapping (e.g. "Junior Consultant" with no
  matching Literal value) and missing-location handling are left as
  extraction-prompt judgment calls, not schema changes** — the user
  explicitly decided against adding an `inferred_fields`-style mechanism to
  `ConsultantProfile` to track this, unlike `ProjectBrief`, which already
  has one.

**Assumptions made:**
- Name and address extraction from raw CV text use simple line-based
  heuristics (first Title-Case line not matching a skip-list of known
  section headings; first line containing a Danish street-name suffix like
  "-vej"/"-gade") tuned to this dataset's consistent CV structure, not a
  general name/address parser.
- The six-figure line-up between stage-1 pattern names and stage-2
  classifications assumes the model correctly follows the tool-use schema;
  any span the model fails to classify silently defaults to "benign" rather
  than raising, so a malformed LLM response degrades gracefully instead of
  crashing ingestion.

**Known limitations / what would need to change for production:**
- PII extraction only recognizes Danish phone formats (`+NN...`) and
  Danish street-name suffixes for addresses — a CV from a consultant with a
  non-Danish address or an unconventional phone format would leak an
  address into `ConsultantProfile.location` unredacted (email/phone/
  LinkedIn/GitHub redaction is still general via regex, so those remain
  safe).
- `_extract_full_name`'s skip-list of section headings is a fixed set
  discovered by inspecting this dataset's actual CVs; a CV with a
  differently-worded header (e.g. "Résumé" instead of "Curriculum Vitae")
  could still be picked up as the "name" incorrectly. (An explicit `"Name:"`
  label is now checked first and takes priority over this heuristic when
  present — see Phase 2 (3d) below — but CVs with neither a label nor a
  recognized skip-list heading are still exposed to this limitation.)
- The stage-2 LLM call has no retry/backoff logic; a transient API failure
  during `scan_for_injection` currently propagates as an exception rather
  than being retried or degraded to a lower-confidence heuristic-only
  result.

## Phase 2 (3c) — Extraction (`src/extract.py`)

**What was built:** one Claude tool-use call per CV that turns raw text into
a `ConsultantProfile`, wired into `src/trust.py`'s `scan_for_injection` and
`separate_pii`, run as a batch script over all 21 ingested CVs. Output:
`data/processed/profiles.json` (21/21 succeeded, 0 failures) and
`data/processed/personal_data.json`.

**Key design decisions and the alternative rejected:**
- **Forced tool-use (`tool_choice={"type": "tool", ...}`) with the tool's
  `input_schema` set to a Pydantic model's `model_json_schema()`**, the same
  pattern `src/trust.py` already uses for stage-2 classification — rejected
  asking for raw JSON in the completion text and `json.loads`-ing it. Tool
  use structurally guarantees syntactically valid JSON and keeps the whole
  codebase consistent on one call pattern; it still doesn't guarantee
  *schema* conformance (e.g. an invalid `seniority` literal), which is what
  the Pydantic-validate-then-retry-once logic is for.
- **`consultant_id` is generated deterministically in Python from the CV's
  filename, never returned by the model** — a separate `_ExtractedProfile`
  schema (identical to `ConsultantProfile` minus `consultant_id`) is what the
  tool call actually returns; `consultant_id` is stitched in afterward. ID
  assignment is bookkeeping the pipeline must be able to trust completely,
  not something worth an LLM's judgement call.
- **Every extraction call also independently runs `trust.scan_for_injection`
  on the same raw text, and the results are merged into `trust_flags`
  alongside whatever the extraction model self-reports** — rejected relying
  on the extraction prompt's self-reported flags alone. The two mechanisms
  catch different things by design (a regex+classifier pipeline tuned
  specifically for injection patterns vs. a general extraction model told to
  notice anything instruction-shaped as a side task), so both are kept and
  deduped rather than picking one.
- **Retry-once-with-the-Pydantic-error-appended, then give up and write to
  `data/processed/failed/`** — rejected either failing the whole batch on
  one bad CV or retrying indefinitely. One retry recovers the common case
  (a one-off bad enum value or empty evidence string) cheaply; a CV that
  still fails after seeing its own validation error is more likely a genuine
  edge case worth a human look than something a third attempt would fix.

**Assumptions made:**
- The retry sends a fresh single-turn call (full CV text again, plus the
  validation error appended) rather than continuing the original
  conversation with the model's own bad tool call replayed back to it —
  simpler to implement correctly for a PoC, at the cost of the model not
  seeing exactly what it said wrong, only the resulting error message.
- Evidence strings are required to be non-empty (schema-enforced,
  `Skill.evidence` `min_length=1`) and the prompt instructs verbatim
  quoting, but nothing checks at runtime that the quote is actually a
  substring of the CV text — CVs are messy enough (line wraps, resolved
  ligature/bullet glyphs from `src/ingest.py`) that a strict substring check
  would likely reject as much legitimate evidence as it would catch
  fabricated evidence.
- `consultant_id` slugs are derived from the source filename
  (lowercased, non-alphanumeric runs collapsed to `_`) with a collision
  counter as a safety net; verified unique and readable across this
  dataset's 21 filenames, but not tested against filenames that could
  legitimately collide after slugification.

**Known limitations / what would need to change for production:**
- All 21 CVs in this dataset succeeded on the first or second attempt in the
  actual run, so the `data/processed/failed/` path is implemented but
  unexercised against a real validation failure from this batch — its
  behavior is only verified against synthetic invalid responses in
  `tests/test_extract.py`.
- No retry/backoff for transient API failures (same limitation noted for
  `src/trust.py`'s stage-2 call) — a network blip during extraction
  currently propagates as an exception and aborts the whole batch run rather
  than being retried or resumed from where it left off.
- `tests/test_extract.py` exercises the retry/validation/merge logic against
  a fake Anthropic client rather than the live API, to keep the test suite
  fast and free; it does not verify that the real model's tool-use output
  actually matches expectations beyond what the one live batch run above
  showed.
- The live run surfaced a genuine planted test case: `cvs-ppt-format.pptx`
  slide 2 has a header naming "Christina Hansen" but a narrative body that
  refers to her as "Sara" throughout. The model correctly caught the
  mismatch and dropped `extraction_confidence` to 0.55 (the lowest of all 21
  CVs), and `trust.py`'s regex-based `_extract_full_name` correctly pulled
  "Christina Hansen" into `personal_data.json`. But the *text* of the
  self-reported `trust_flags` entry hallucinated the quoted header name as
  `cvs_ppt_format_slide2` -- the Python-assigned `consultant_id`, a string
  the model was never shown -- instead of "Christina Hansen". The underlying
  signal (name mismatch -> lower confidence) was correct; only the
  human-readable quote inside the flag string was fabricated. This is
  contained to `trust_flags` (advisory, `list[str]`, nothing downstream
  parses it structurally) and doesn't affect `full_name`, `current_role`, or
  any other field. It's a concrete illustration of why evidence strings
  aren't verified against raw_text at runtime (noted above) and, by
  extension, why nothing downstream should treat `trust_flags` text as a
  verified quote rather than an LLM's best-effort explanation.

## Phase 2 (3d) — Availability (`src/availability.py`)

**What was built:** a synthetic weekly-availability generator that assigns
every consultant in `data/processed/profiles.json` a bench status
(`available` / `partly_booked` / `fully_booked`), free days per week, a next-
free date, and (if booked) a current project, writing the result to
`data/processed/availability.csv`. No LLM call involved — deterministic
Python, matching this project's rule that anything downstream of extraction
is plain code, not model judgement.

**Key design decisions and the alternative rejected:**
- **Turn the target 25/45/30 split into exact integer counts via
  largest-remainder rounding, then shuffle with a seeded RNG** — rejected
  drawing each consultant's status independently (e.g. `rng.choices` with
  weights). At n=21, independent draws can land noticeably off the target
  split on any given run; computing exact counts first and only using the
  RNG to decide *which* consultant gets which status keeps the aggregate
  distribution stable while still varying who lands where.
- **Deterministically pick one strong data/AI consultant (by role-title
  keyword, then ranked by seniority/skill-count/years) and force them to
  `fully_booked`**, outside the random pool — rejected leaving this to
  chance. The case brief specifically needs a "great-fit candidate who isn't
  actually available" example for the trade-off view; leaving it to the RNG
  risked a demo run where no strong data/AI candidate happened to land on
  `fully_booked`.
- **`next_free_date` is seeded only as an *offset* from real "today," not
  a fixed calendar date** — rejected freezing the whole dataset (including
  dates) to one fixed day. A demo should show availability relative to
  whenever it's actually run; only the RNG choices (status assignment,
  free-days values, project picks, week offsets) need to be reproducible,
  not the absolute date.

**Assumptions made:**
- All availability values are fabricated — there's no real staffing/CRM
  system behind this PoC. Project names are drawn from a small fictional
  pool.
- `available`/`partly_booked` consultants have `next_free_date` = today
  (they already have some open capacity now); only `fully_booked`
  consultants get a future date, `free_days_per_week` = 0. This treats
  "next free" as "next date with *any* open capacity," not "next date fully
  unbooked" — a defensible but not the only reasonable reading of the
  column name.

**Known limitations / what would need to change for production:**
- `tests/test_availability.py` skips itself when
  `data/processed/profiles.json` doesn't exist yet (it isn't committed —
  see below), matching the same pattern `tests/test_ingest.py` and
  `tests/test_trust.py` already use for `data/raw/`.
- `data/processed/` was added to `.gitignore` at this point (it wasn't
  before) because `personal_data.json` holds PII-shaped output and the
  whole directory is fully reproducible from `data/raw/` via
  `python -m src.extract` followed by `python -m src.availability` — it was
  never meant to be a committed artifact.

**Bugfix discovered while reviewing this phase's output (`src/trust.py`):**
Manually reviewing `availability.csv` joined against `personal_data.json`
surfaced that two consultants (`cv3`, `cv5`) had `full_name: "Deployment
History"` — a section heading, not a person. Root cause: `CV3.docx` and
`CV5.docx` put the name under a "Personal Information" section as
`"Name: Mikkel T. Rasmussen"`; the mid-initial period means that line can't
match `_extract_full_name`'s bare Title-Case-line heuristic, so it fell
through to the first *unrelated* Title-Case heading further down the
document instead. Fix: check for an explicit `"Name:"` label first (verified
against all 21 CVs to appear in exactly 3 places, no false-positive risk),
falling back to the existing bare-line heuristic only when no label is
present — rejected just adding "deployment history" to the skip-list, since
that would only have delayed the same failure mode to the next
unrelated heading a future CV happens to use. Regression tests added in
`tests/test_trust.py` against the real CV3/CV5 files. `data/processed/`
was regenerated end to end (`extract` then `availability`) after the fix.

## Phase 3 — Matching engine (`src/match.py`, `src/graph.py`)

**What was built:** the four-stage matching pipeline — deterministic hard
filter, hybrid BM25 + semantic retrieval, one LLM re-rank call per role, and
deterministic team assembly producing a recommended team plus two
alternatives (earliest start, lowest cost) — plus `src/graph.py`'s
co-delivery adjacency graph. Smoke-tested end to end against the real
21-CV/`availability.csv` dataset (no live LLM call): the hard filter funnel
narrowed 21 → 15 → 14 → 7 → 7 on a realistic brief, the co-delivery graph
found 4 real shared-project edges, and retrieval for a "Data Scientist" role
surfaced data-scientist/ML-engineer/analytics-consultant profiles first.

**Key design decisions and the alternative rejected:**
- **Added `ProjectBrief.required_language: str | None`.** The phase-3 brief
  requires a hard language filter, but the schema had no field to carry it.
  This is a deliberate, narrowly-scoped addition — not the kind of
  extraction-ambiguity schema change the user previously rejected (see
  phase 2's "Ambiguous seniority-tier mapping" decision above); it adds a
  genuinely missing piece of required data, not a mechanism for resolving
  judgment calls.
- **Co-delivery edges (`src/graph.py`) use only exact (case/whitespace-
  normalised) `Project.title` matches, not the brief's requested "same
  client+year" criterion** — rejected because `Project` has no client
  field, and the next-best proxy, same industry + same `year_start`, was
  tested against the real dataset and found to group 20 unrelated
  engagements as "shared" purely for happening in the same industry the
  same year. That would fabricate co-delivery bonuses between consultants
  who never actually worked together, which is worse than under-counting.
  Exact-title matching found 4 real edges in the dataset (including CV1/CV5
  on "Complexity Reduction in Portfolio and Operations") at the cost of
  missing pairs who describe the same engagement in different words (e.g.
  "...Rollout" vs no suffix) — a real but bounded gap, documented in
  `src/graph.py`'s docstring.
- **"Lowest cost" team assembly uses a candidate's own seniority tier as a
  cost proxy** — rejected leaving cost unimplemented, since nothing in this
  project (`ConsultantProfile`, `PersonalData`, `availability.csv`) has a
  rate-card or billing-rate field. Unlike the rejected industry+year
  co-delivery proxy, seniority-as-cost is a defensible, close-to-universal
  assumption in consulting, not a noisy one.
- **Stage 1's must-have-skill and required-language filters use exact
  (case-insensitive, trimmed) string matching, not fuzzy matching** —
  rejected substring/fuzzy matching for these two hard filters specifically,
  because a false-positive pass on a *hard* filter (recommending someone who
  doesn't actually have the required skill) is a worse failure mode than a
  false negative (a real match phrased differently gets dropped and is
  invisible in the funnel, but at least never wrongly recommended). Location
  matching is looser (substring either direction) because it's a softer,
  more forgiving constraint by nature (e.g. "Copenhagen" vs "Copenhagen,
  Denmark").
- **"Availability in the project window" only checks that
  `next_free_date <= parsed start_date`, not any end-of-window constraint**
  — rejected trying to model `duration_weeks` against availability, because
  `availability.csv` (src/availability.py) only stores one current bench
  status and one next-free date per consultant, not a future calendar. There
  is nothing to check an end date against.
- **A small hand-rolled parser (`_parse_start_date`) handles ASAP-style
  synonyms, ISO dates, "Q<n> YYYY", and "<Month> YYYY"; anything else is
  treated as "no window constraint" (the filter is skipped, not applied)**
  — rejected either crashing on unparseable text or defaulting to "today"
  for anything unrecognised. A hard filter that silently passes everyone on
  unparseable input is safer for a demo than one that silently drops
  everyone; both are documented, but "pass" was judged the less surprising
  failure mode for a staffing recommendation tool.
- **`hybrid_retrieve` treats an all-empty BM25 corpus (no candidate has any
  skill/project/industry/certification text) as "zero keyword signal for
  everyone" rather than crashing** — discovered because `BM25Okapi` itself
  divides by zero building its idf table when its corpus has no tokens.
  Real profiles almost always have skills, but the failure mode (crashing
  stage 2 entirely) was worse than the fix (skip BM25 scoring, rely on
  semantic score alone) was risky.
- **Stage 3's `rerank_role` has no retry-on-validation-failure**, unlike
  `src/extract.py`'s extraction call — kept consistent with `src/trust.py`'s
  stage-2 classification call, which has the same limitation for the same
  reason (simplicity for a PoC; a transient bad response currently
  propagates as an exception).
- **The two alternative teams (earliest start, lowest cost) only optimise
  within each role's top-5 LLM-fit-scored candidates
  (`_ALT_TEAM_CANDIDATE_POOL`), and skip the recommended team's
  complementarity/co-delivery/booking adjustments entirely** — rejected
  either optimising across the full shortlist (risks recommending a
  genuinely poor-fit candidate purely for being cheap or free sooner) or
  applying the same adjustment stack (would blur what each alternative team
  is actually optimised for). Restricting to a fit-scored top-5 keeps every
  alternative a *plausible* team while still letting the one named
  dimension (start date, cost) dominate the choice within it.

**Assumptions made:**
- All score-weighting constants (BM25/semantic 40/60, skill-overlap
  penalty, co-delivery bonus, booking penalty, alt-team pool size, retrieval
  top-k) are tunable PoC values flagged inline in `src/match.py`, not
  validated against any ground truth — there is no labelled "good team"
  dataset to validate against in this PoC.
- Stage 4's role fill order sorts by `(-seniority_rank, surviving_candidate_
  pool_size)` — most senior role first, ties broken by scarcity (fewest
  ranked candidates) — a literal reading of the brief's "most senior/most
  constrained role first," not a weighted combination of the two.

**Known limitations / what would need to change for production:**
- The co-delivery graph under-counts real co-delivery (exact-title matching
  only, no client field to match on) — see above.
- No cost/rate-card data exists anywhere in this project; "lowest cost" is a
  seniority-tier proxy, not a real cost calculation.
- `rerank_role` has no retry logic; a transient API failure or a malformed
  tool-use response aborts that role's ranking (and therefore team
  assembly) entirely.
- `tests/test_match.py` and `tests/test_graph.py` use fake Anthropic clients
  and fake embedding functions throughout, matching this project's existing
  testing convention (see phase 2's `tests/test_extract.py`) — they verify
  the pipeline's logic, not that the real `claude-sonnet-4-6` model's
  tool-use output matches expectations, or that `all-MiniLM-L6-v2`'s actual
  embeddings produce good semantic rankings. A real-data smoke test (hard
  filter + retrieval only, no live LLM call, described above) exercised the
  non-LLM stages against the actual dataset; stage 3's live LLM call and
  full `match()` orchestration were not exercised against the real API in
  this phase.

### Phase 3 follow-ups from live demo testing

Running real briefs end to end against the actual dataset and API (not just
the fake-client test suite) surfaced four gaps, each fixed the same session:

**1. `availability_tradeoffs` (src/schema.py, src/match.py).** Testing a
"Cloud migration, Nordic retailer, 12 weeks" brief with `start_date="ASAP"`
surfaced that the dataset's best-fit cloud architect, and two other strong
candidates, were silently missing from every team -- all three were
`fully_booked` and failed the stage-1 availability filter before the LLM
ever saw them. Nothing in `MatchResult` said so; the user had to notice
their absence and manually cross-reference `availability.csv` to find out
why. Added `MatchResult.availability_tradeoffs`: every candidate who passes
language/must-have-skills/location but fails only availability, with their
`next_free_date` and how many days after the requested start that is.
**Rejected:** generalising this into a full counterfactual across every
hard filter ("who'd qualify if we dropped the language requirement") --
that's CLAUDE.md's repo layout already scoping "counterfactual" to
`src/explain.py`, a later phase; this is the one tradeoff cheap enough to
compute here with data `hard_filter` already touches, and the one a user
hits first in practice.

**2. `HF_HUB_OFFLINE=1` (src/match.py `_get_embedder`).** Every embedder
load printed a Hugging Face Hub rate-limit warning, because
`sentence-transformers` checks Hub for cache freshness on every load even
when the model is already cached locally. Not a security or correctness
issue, but a live network dependency during an in-person case-interview
demo is a real risk (flaky wifi, corporate proxy, Hub outage) for zero
benefit once the model is cached. Set `HF_HUB_OFFLINE=1` via
`os.environ.setdefault` before the model loads. **Rejected:** an `HF_TOKEN`
instead -- a token raises rate limits but doesn't remove the live network
call, so it doesn't fix the actual risk (demo-day flakiness); it also adds
a secret to manage for no corresponding benefit in a single-laptop demo
context. **Trade-off accepted:** a machine that has never loaded the model
before must run once with the flag unset (or `HF_HUB_OFFLINE=0`) to seed
the cache -- irrelevant for this demo's one laptop, real for a fresh
machine or CI.

**3. `StaffingGap` / `Team.gaps` (src/schema.py, src/match.py).** Stress-
testing a deliberately impossible brief ("Quantum Cryptographer, starting
Monday" -- no one in a 21-person management/tech consulting dataset is a
quantum specialist) showed the pipeline behaving correctly but silently: it
returned a real person (the closest available AI/ML engineer) with an
honest fit_score of 6-7/100 and a concern explicitly stating the role's
core requirements were unmet -- but nothing structurally flagged that this
"recommendation" shouldn't be trusted. A fit_score of 7 sitting quietly
inside an otherwise normal-looking team card is easy to miss. Added
`StaffingGap` (`role_title`, `reason: "understaffed" | "low_confidence_fit"`,
`detail`, optional `consultant_id`) and `Team.gaps`, computed by
`_staffing_gaps()` from each member's raw `fit_score` (not
`assembly_score` -- a low fit is a genuine competence gap; a low assembly
score can just mean a fine candidate was penalised for redundant skills or
thin availability, a different kind of problem). Threshold
`_LOW_CONFIDENCE_FIT_THRESHOLD = 30.0` is a tunable PoC cutoff, flagged
inline, not validated against any labelled "this recommendation was
actually bad" dataset -- none exists to validate against here. Also catches
understaffed roles (fewer members than `role.count` demanded), reusing the
same mechanism. Verified against the live quantum-cryptographer brief: all
three teams now carry `gaps: [{reason: "low_confidence_fit", ...}]`
alongside the same fit-7 suggestion, instead of the suggestion appearing
unqualified.

**4. Concurrent `rerank_role` calls (src/match.py `match()`).** Profiling a
real brief found `rerank_role` (stage 3's live Claude call) taking 60-80s
per role -- driven by output length (15 candidates x fit_score + 3 cited
reasons + concern is ~3,000-4,000 generated tokens per call), not input
size or embedding cost, which together stayed under 1 second total.
Sequential, a 2-role brief spent ~150s in LLM calls alone, making the tool
unusable live in a case interview; a brief with more roles would scale
linearly and get worse. Since each role's ranking is independent of every
other role's, `match()` now dispatches all `rerank_role` calls concurrently
via `ThreadPoolExecutor` (threads, not asyncio, since each call is a single
blocking HTTP request and the client library isn't async) and resolves one
shared `anthropic.Anthropic()` client up front rather than letting each
thread construct its own. **Rejected:** reducing `_RETRIEVAL_TOP_K` or
trimming the candidate payload sent to the LLM to cut latency -- both would
reduce evidence quality or shortlist breadth to solve a problem
concurrency solves for free with no quality trade-off. Verified on the live
cloud-migration brief: wall time dropped from ~150s of sequential LLM time
to 76.1s total (bounded by the slower of the two concurrent calls, not
their sum) with structurally identical output (same funnel, same
availability tradeoffs, same core three-person team) -- individual
fit_scores and which role each person landed in shifted between runs, which
is expected LLM run-to-run stochasticity on independent calls, not a
regression from parallelising. **Known limitation:** a role's API failure
still aborts the whole `match()` call when `future.result()` re-raises it --
same failure mode the sequential version already had, not a new one, and
still no retry/backoff (consistent with `rerank_role`'s existing documented
limitation).

**What this means for the 2,000-CV scaling question:** the funnel design
(hard filter -> fixed top-15 retrieval -> LLM re-rank) already decouples
stage 3's cost from total corpus size -- `rerank_role` always scores a
fixed-size shortlist regardless of whether the candidate pool is 21 or
2,000, so growing the dataset alone doesn't slow the expensive stage down.
What would need to change at that scale is stage 2: `hybrid_retrieve`
currently recomputes every surviving candidate's embedding from scratch on
every call (cheap at 21 candidates, wasteful at 2,000) and rebuilds the
BM25 index per role even though the candidate pool doesn't change between
roles within one `match()` call. Neither is fixed yet -- the fix is
precomputing and caching each profile's embedding once (e.g. at extraction
time) so retrieval only embeds the short role-query text live, and
building the BM25 index once per `match()` call instead of once per role.
A real vector index (FAISS etc.) isn't needed at 2,000 rows -- a cached
numpy matrix with `sklearn.cosine_similarity` stays fast well beyond that.
What scales worst is role count, not CV count: each additional distinct
role adds another ~60-80s call: concurrency (this phase) already turns that
from "sum of all roles' latency" into "roughly one call's worth," bounded
by whatever request concurrency the Anthropic API allows.

## Phase 4 (part 1) — Explanation (`src/explain.py`)

**What was built:** `build_match_card` / `build_match_cards_for_team`, which turn one
`TeamMember` (from `src/match.py`'s stage 4) into a `MatchCard`: an overall score, a
deterministic five-component breakdown (skill fit, seniority fit, availability,
industry experience, team chemistry), up to three verbatim CV evidence quotes with a
best-effort project attribution, a trust badge (verified / flagged / unverified
claims), and a counterfactual against the next-best candidate who wasn't picked. Zero
new LLM calls anywhere in this file -- everything is either reused directly from
`src/match.py`'s stage-3 output or computed by plain Python over structured data
already sitting in `ConsultantProfile`, `availability.csv`, and the co-delivery
graph, per the phase-4 brief's explicit requirement that the counterfactual come from
score-breakdown deltas, not an extra call.

**Coverage check against phase 3 before building (per the phase-4 brief's own
instruction to skip what's already covered):** overall score and the per-candidate
concern were already on `TeamMember` (`fit_score`, `concern`) and are reused, not
rebuilt. Everything else -- the five-component breakdown, verbatim evidence
selection, the trust badge, and the counterfactual -- had no prior implementation:
`TeamMember.adjustments` is a different three-way decomposition for team *assembly*
(skill-overlap penalty / co-delivery bonus / booking penalty), not a candidate-facing
explanation; `TeamMember.reasons` are LLM prose *about* evidence, not verbatim CV
quotes; and the data a counterfactual needs (every stage-3-scored candidate per role,
not just the winner) was computed inside `match()` but discarded once teams were
assembled.

**Key design decisions and the alternative rejected:**
- **`MatchResult` gained a `role_rankings` field** (every `RoleFitScore` per role,
  not just who was picked) -- rejected either recomputing rankings inside
  `src/explain.py` (would re-trigger LLM calls stage 3 already paid for, and violate
  the "no extra LLM call" requirement in spirit even though the call itself isn't
  what powers the counterfactual) or threading `role_rankings` through as a bare
  parameter instead of a schema field (loses the "one MatchResult carries everything
  a caller needs" property `funnel` and `availability_tradeoffs` already established
  in phase 3).
- **The five breakdown components are a separately-computed, deterministic
  *explanation* of fit, not a formula whose weighted sum reproduces `fit_score`** --
  rejected trying to reverse-engineer weights that would make the five components
  sum to the LLM's holistic score. `fit_score` is a single judgement call the model
  makes over everything at once (including things these five proxies can't see, like
  how compellingly a project's impact reads); forcing an exact arithmetic
  relationship would mean either the proxies stop being honest independent measures
  or the breakdown stops matching the score, and lying about either is worse than a
  breakdown that's presented as a complementary lens rather than a decomposition.
  Verified this divergence is real and useful, not just theoretical, on live data:
  Katrine Pedersen scored `fit_score=85` (the LLM correctly read "Predictive
  Modeling" as semantically close enough to the role's "Predictive Analytics"
  requirement) but `skill_fit=33.3` (the deterministic exact-match component
  correctly does *not* treat those as the same string) -- both numbers are honest,
  and showing both is more useful than picking one.
- **`team_chemistry` is recomputed fresh against a team's final roster, not reused
  from `TeamMember.adjustments`** -- rejected reusing the stored field because it
  only exists for the "recommended" team (the two alternative teams leave
  `adjustments` empty, per phase 3's design) and is itself an order-dependent
  snapshot from greedy assembly (computed against whoever had already been picked
  *at that point*), not a description of the finished team. Recomputing gives every
  team variant, and every card -- including the counterfactual's alternative
  candidate, who was never actually assembled into anything -- the same
  well-defined, order-independent answer.
- **Evidence quotes are deduped by exact quote text, not just picked by
  required-skill-match-then-confidence** -- discovered necessary from real data, not
  anticipated in the initial design: several CVs in the dataset quote a whole
  "Technical Skills:" list line as the verbatim evidence for *every* skill on that
  line (a legitimate quote per `src/extract.py`'s rules when nothing more specific
  exists per-skill). Without deduping, a card's "top three evidence sentences" could
  be the same sentence three times, which defeats the stated purpose. Fixed by
  skipping any skill whose evidence text was already selected and continuing down
  the ranked list, verified against Katrine Pedersen's real profile (10 of her 24
  skills shared one identical evidence string).
- **skill_fit and industry_experience reuse phase 3's exact-match (case-insensitive,
  trimmed) convention, not fuzzy matching** -- same rejected alternative and same
  reasoning as `src/match.py`'s hard filter (see phase 3): a false-positive "good
  fit" signal from fuzzy matching is worse than an honest low score on a real match
  phrased differently.
- **The counterfactual's "next-best candidate" is picked by highest `fit_score`
  among the remainder, not by any breakdown component** -- rejected picking by, say,
  highest `skill_fit`, since `fit_score` is what stage 3's ranking is actually
  sorted by and what a user would recognise as "the next name down the list."
  Candidates already staffed elsewhere on the *same* team are excluded, since
  swapping in someone already committed to another role isn't an actionable trade.

**Assumptions made:**
- All five breakdown-component weights (seniority-mismatch penalty per tier,
  availability scoring by status, industry match/no-match scores, team-chemistry
  baseline/bonus/penalty) are tunable PoC constants flagged inline in
  `src/explain.py`, not validated against any labelled "this explanation was
  actually accurate" dataset -- none exists here, same caveat as phase 3's assembly
  weights.
- Project attribution for an evidence quote (`EvidenceQuote.project_title`) is a
  best-effort guess -- the first project whose `tech` list names the skill, or
  `None` if none does -- because `src/schema.py`'s `Skill` and `Project` models have
  no structural link between them. Verified against real data this produces `None`
  reasonably often (e.g. a skill like "Time Series Analysis" that appears in a
  profile's top skills list but not literally in any project's `tech` list), which
  is an honest "couldn't attribute this" rather than a guess dressed up as certain.
- The trust badge only distinguishes three states computed from data already
  produced upstream (`ConsultantProfile.trust_flags`, `extraction_confidence`) -- it
  does not re-verify that any evidence quote is an actual substring of the raw CV
  text, which is the same known limitation `src/extract.py` and `src/trust.py`
  already document (see phase 2).

**Known limitations / what would need to change for production:**
- No UI yet (`app.py` per `CLAUDE.md`'s repo layout is still unbuilt) -- match cards
  were verified against real data via a one-off script printing them to the
  terminal, not through the Streamlit app this phase's name ("Explanation & UI")
  implies is still to come.
- The five breakdown weights are hand-picked, not tuned against any ground truth,
  same limitation as phase 3's assembly weights.
- Evidence-quote deduplication only catches *exact* duplicate strings -- two
  differently-worded evidence quotes that both describe the same underlying claim
  would still both be shown, since there's no semantic dedup, only literal.

### Phase 4 (part 1) follow-ups from live demo testing

Testing real briefs (including one deliberately aimed at CV4, the CV with a
confirmed prompt-injection attempt -- see phase 2) surfaced three more gaps, each
fixed the same session:

**1. `Counterfactual.trust_badge` (src/schema.py, src/explain.py).** Running a real
"ERP change management, manufacturing" brief, one candidate's counterfactual pointed
at CV4 with the summary "would improve every component... with no clear downside" --
true of the score breakdown, but silent about CV4's flagged CV. `Counterfactual` had
no `trust_badge` field, so the one signal that should stop someone from acting on a
swap suggestion could never appear on it. Added `trust_badge` to `Counterfactual`,
and `_summarize_swap` now appends a trust caveat whenever the alternative isn't
"verified" (`_TRUST_CAVEATS`). Verified against the exact real scenario: the summary
now reads "...with no clear downside. Note: this candidate's CV has a flagged trust
issue -- see their trust badge before acting on this."

**2. `MatchCard.availability_alternative` (src/schema.py, src/explain.py) -- the
feature this part was built to add.** `MatchResult.availability_tradeoffs` (phase 3)
and `Counterfactual` (this phase) were completely disconnected: a `Counterfactual` is
only ever built from `role_ranking`, which only contains candidates who survived
`hard_filter` -- so a candidate excluded for availability alone, however strong a fit
they'd score, could never appear as a counterfactual, only in the separate, unscored,
brief-wide tradeoffs list with no role attached. Added `AvailabilityAlternative` (a
deliberately different shape from `Counterfactual`, not a `Counterfactual` with an
optional `fit_score` -- these candidates never went through stage 3, so there
genuinely is no `fit_score` to report) and `build_availability_alternative`, which
deterministically breakdown-scores every tradeoff candidate against a role and
surfaces the best one, but only when their future-fit rank (`skill_fit`,
`seniority_fit`, `industry_experience`, `team_chemistry` -- `availability` itself is
deliberately excluded from the ranking key, since every tradeoff candidate's
availability score is low by construction and would only add a constant offset, not
signal) genuinely exceeds the pick's -- a card showing "here's someone worse who's
also unavailable" on every role would be noise, not signal. Verified against real
data: Louise Damgaard (S&OP/Demand Planning, Consumer Goods/FMCG industry, but
`fully_booked`) correctly surfaces on Ida Mørk's real Forecasting Analyst card
("+55 industry experience, -20 availability... not free until 6 days after your
requested start"). **Measured, not assumed, before building:** `compute_breakdown`
costs ~11 microseconds/call: a realistic brief (6 tradeoff candidates x 2 roles) adds
~0.13ms; a pathological 2,000-CV worst case (200 tradeoff candidates x 5 roles) adds
~11ms -- versus the 60-80 *seconds* per `rerank_role` call that actually dominates
`match()`'s wall time (phase 3). No LLM call, no embedding call -- pure Python over
data already in memory.

**3. `classify_trust` was over-flagging roughly half the real dataset -- found and
fixed, not part of the original plan.** The original implementation treated any
non-empty `trust_flags` as `"flagged"`. Checked against all 21 real profiles: ~10
would have shown "flagged", including CVs whose *only* trust_flags entry explicitly
cleared them -- e.g. `"No prompt injection or adversarial text detected in the
document."`, which literally contains the word "injection" while saying the
opposite. `src/extract.py`'s self-reported `trust_flags` entries are free text with
no consistent structure (verified: they range from genuine concerns to entirely
benign extraction caveats like "no certifications section found" or "project years
inferred from employment dates", with no reliable way to distinguish them from
wording alone) -- only the *scanned* entries `_merge_trust_flags` produces have a
guaranteed structure (`"injection (...)"` / `"promotional_language (...)"`, and are
already filtered to exclude "benign" classifications before merging). **Fix:**
`classify_trust` now only fires `"flagged"` on that structured prefix, and uses
`extraction_confidence` alone (not trust_flags non-emptiness) for
`"unverified_claims"`, since confidence is a consistent numeric signal computed the
same way for every profile. **Rejected:** a keyword search (e.g. "injection" or
"promotional" anywhere in the flag text) as a middle-ground fix -- rejected because
it still would have flagged the self-reported "No prompt injection detected"
entries, which contain the trigger word while asserting the opposite finding.
**Verified against the full real dataset, before vs. after:** ~10/21 "flagged" before
→ exactly 2/21 non-"verified" after (`cv4`: flagged, the real confirmed injection;
`cvs_ppt_format_slide2`: unverified_claims, the real name-mismatch/low-confidence
case from phase 2 -- extraction_confidence 0.55, the lowest in the dataset), with
every other profile now correctly "verified".

**Assumptions made (this part):**
- `AvailabilityAlternative`'s "best of the tradeoffs list" ranking is a plain mean of
  four breakdown components, not a validated weighting -- same caveat as every other
  scoring constant in `src/match.py` and `src/explain.py`.
- The "only surface if strictly better" comparison uses no margin (any positive
  difference in future-fit rank qualifies) -- a small margin might reduce noise from
  near-tied comparisons, but wasn't judged necessary without evidence of it being a
  real problem in practice.

**Known limitations (this part):**
- `classify_trust`'s three-tier design still can't recover a genuine self-reported-
  only finding that `src/trust.py`'s stage-1 regex patterns miss entirely (no
  structured flag would ever be produced for it) -- mitigated but not eliminated by
  stage 1 being deliberately broad/recall-oriented (see phase 2), not eliminated by
  design.
- `AvailabilityAlternative` only compares a tradeoff candidate against the single
  role a match card is already about -- it does not search across every role in the
  brief for the tradeoff candidate's single best-fitting role, so a tradeoff
  candidate who'd be a mediocre fit for the role being explained but an excellent
  fit for a *different* role in the same brief will never surface anywhere.

## Phase 4 (part 2) — Brief intake (`src/brief.py`) and the UI (`app.py`)

**What was built:** `src/brief.py` (`parse_brief`), the last missing piece of CLAUDE.md's
repo layout: one LLM tool-use call, same pattern as `src/match.py`'s `rerank_role`,
turning a free-text staffing request into a validated `ProjectBrief` -- including
proposing a sensible `roles_needed` staffing plan when the request doesn't spell
that out, which real requests almost never do. And `app.py`: the Streamlit UI --
free-text input, three example briefs chosen for three different real outcomes
seen in testing, expandable evidence-backed match cards, two alternative teams,
always-visible team-availability and consultant-network views, session history,
and a live-ticking elapsed-time clock. This is the first time the full chain (free
text -> `brief.py` -> `match.py` -> `explain.py` -> on-screen) has run end to end
without a hand-built `ProjectBrief` object anywhere in the loop.

**Key design decisions and the alternative rejected:**
- **`must_have_skills` guidance in `brief.py`'s system prompt had to be tightened
  after it failed its own first real test.** The first version told the model to
  "keep it conservative"; live-tested against "ERP change management,
  manufacturing, 24 weeks from September", the model still proposed
  `must_have_skills=["ERP", "change management"]` -- verified against the real
  dataset this would have silently eliminated Nikolaj Friis, the single strongest
  real candidate for that exact brief (96/100 `fit_score` in phase-3 testing),
  because his actual extracted skills are phrased "SAP S/4HANA" and "Change
  Management Strategy", neither of which equals the generic terms an LLM naturally
  reaches for. Rejected leaving the soft "be conservative" instruction as-is; fixed
  by stating the exact failure mode explicitly in the prompt rather than trusting
  the model to infer the risk. Re-verified live after the fix: `must_have_skills`
  came back empty, and the full chain (`brief.py` -> `match.py` -> `explain.py`)
  correctly surfaced Nikolaj Friis as the top pick (92/100).
- **`brief.py` does not wrap the request text in `<document>` tags with CLAUDE.md
  rule 3's "untrusted data" framing, unlike every CV-handling prompt in this
  project** -- rejected applying rule 3 uniformly regardless of who authored the
  text. Rule 3 exists because a CV is written by a third party a consultant doesn't
  control; a staffing brief is typed by the tool's own operator describing their
  own request, a different trust boundary. The system prompt still tells the model
  to treat the text as a request to structure, as basic hygiene, short of the full
  untrusted-data ceremony.
- **The "Find team" click runs `parse_brief` + `match` in a background thread while
  the main thread polls every 0.5s and updates a live elapsed-time display** --
  rejected `st.status` alone (used in an earlier draft), which can only report
  progress at fixed checkpoints the main script thread reaches, not a genuinely
  ticking clock, since the thread is blocked for the full 60-150s duration of the
  match call either way. The background thread never calls any `st.*` function
  itself (Streamlit widgets are tied to the script-run thread) -- it only writes to
  a plain dict the polling loop reads, which is also where real per-step timing
  data comes from for the "time spent per step" chart.
- **Both `parse_brief` and `match` are wrapped in their own `@st.cache_data`
  function, keyed on the brief text (or the parsed brief's own JSON) plus today's
  date, deliberately for demo stability, not just speed** -- fit scores and role
  assignments have been observed to vary between identical runs of the same brief
  across this whole project (LLM non-determinism); resubmitting an already-run
  brief should show the same result, not re-roll it live in front of an
  interviewer. The match-step cache function deliberately lets exceptions
  propagate *uncached* though, unlike the parse-step one -- `rerank_role` has no
  retry logic, so a transient API failure must stay retryable on the next click,
  not get permanently remembered as a failure the way a genuine two-attempt schema
  validation failure (parse-step) reasonably can be.
- **Team availability and the consultant co-delivery network are always-visible
  expanders, not gated behind a match result** -- moved here after direct user
  feedback that they wanted these visible from the start, independent of running a
  brief. When a result does exist, the network diagram highlights the current
  recommended team's members in a different colour from everyone else with
  shared-project history.
- **The co-delivery network is a hand-rolled inline-SVG circular-layout diagram
  (`network_layout` + `render_co_delivery_network`), not graphviz or networkx** --
  checked first and neither package nor the system Graphviz binary is installed;
  rejected adding either as a new dependency when N is small (a few dozen nodes at
  most) and a self-contained SVG needs nothing external at all, matching the same
  reasoning as `HF_HUB_OFFLINE` in `src/match.py`: nothing in a live demo should
  depend on a resource that could fail to load.
- **Score breakdown and both funnels (candidate-survival, time-per-step) use
  `st.bar_chart`, bundled with Streamlit** -- rejected a separate plotting library
  for a PoC UI; `st.bar_chart` with `horizontal=True` (confirmed supported in the
  installed Streamlit 1.61) covers both cases without a new dependency.
- **`select_evidence`'s output now carries a `matched_requirement` flag per quote,
  and `app.py` visibly marks unmatched evidence** -- found via a real user report,
  not anticipated: for an "Azure migration" brief, several shortlisted candidates
  (verified: zero Azure/cloud skills anywhere in their profiles) had their cards
  show generic high-confidence skills as unqualified "Evidence", with nothing
  disclosing they had nothing to do with Azure. Root-caused to two stacked causes,
  confirmed against real data: (1) the dataset's actual Azure specialists (Christian
  Enevoldsen, Jonas Kristensen, Mads Egeberg) were all `fully_booked` and excluded
  before scoring even began -- expected behaviour, the exact case
  `AvailabilityAlternative` (phase 4 part 1) exists for; and (2) among who was left,
  retrieval ranked at least one genuinely Azure-skilled candidate (Rasmus Olesen,
  "Azure ML / Azure Functions") *below* three candidates with zero cloud skills at
  all -- a real retrieval-ranking imprecision, documented as a known limitation
  below, not fixed this session (tuning `src/match.py`'s BM25/semantic weights or
  query composition without a labelled ground-truth risks trading one imprecision
  for another). **Rejected** hiding evidence entirely when nothing matches --
  CLAUDE.md rule 4 requires every score to carry evidence, so the fix is
  disclosure (an explicit "general skill, not a direct match" tag, and a banner
  when *no* shown evidence matches) rather than suppression.

**Assumptions made:**
- `app.py`'s session-only history (`st.session_state`), not a database or a file on
  disk, matches the "no login, no database" constraint from the original UI prompt
  -- a live demo runs in one continuous browser session, so history surviving a
  full app restart wasn't judged necessary.
- The header logo is a small inline SVG, not an external image file, for the same
  offline-safety reasoning as `HF_HUB_OFFLINE`.
- The availability view is a colour-coded snapshot table (pandas `Styler`), not a
  calendar -- `availability.csv` only carries one status, one weekly free-day
  count, and one next-free date per consultant, so a real calendar would imply
  day-by-day granularity the data doesn't have.
- Trust badges carry a leading symbol (check / warning triangle / cross) in
  addition to colour, for a colourblind viewer or a washed-out projector during a
  live demo.

**Known limitations / what would need to change for production:**
- **Retrieval-ranking imprecision is real and unfixed**: `hybrid_retrieve`'s 40/60
  BM25/semantic blend can rank a candidate with zero relevant skills above one who
  genuinely has them, verified concretely on a real "Azure migration" query. Fixing
  this properly would need either query/profile-text composition changes or
  weight retuning validated against a labelled relevance dataset this PoC doesn't
  have -- changing it blind risks trading a known imprecision for an unknown one.
- `app.py`'s pure display-data functions are unit-tested; the `st.*` rendering
  functions themselves are not (see the file's own docstring) -- verified instead
  by running the live server and checking the health endpoint plus (this session)
  visual review requests relayed through the user, since no in-session browser
  tool was available.
- `AvailabilityAlternative` still only compares against the single role a card is
  about (phase 4 part 1's limitation, unchanged this part).
- No deployment story exists or was requested -- `app.py` is a local
  `streamlit run` dev server; the URL it serves on is not stable across sessions,
  and there is currently no way to reach it without running that command yourself.

## Post-phase-4 bugfix — multi-city briefs zeroed out the entire candidate pool

**What was built:** Fixed `_passes_location` in `src/match.py`. A live run of the
brief "Enterprise Agility in Automotive, from October 1st, team in Stockholm,
Copenhagen, Oslo or Arhus, with budget of 100k for 3 months" returned zero
candidates for every role. The extracted `brief.location` was
`"Stockholm / Copenhagen / Oslo / Aarhus"` -- a single combined string, reasonable
extraction. `_passes_location` compared that whole string against a candidate's
single-city `"Copenhagen, Denmark"` as one bidirectional substring check, which
can never match a multi-city brief against a single-city profile (neither string
is a substring of the other once the city + ", Denmark"/combined-list punctuation
is accounted for) -- so the location filter silently dropped all 18 survivors of
the earlier stages, not just the three specific consultants the user expected to
see (Jonas Kristensen, Nikolaj Friis, Sebastian Knudsen).

**Key design decision:** Added `_split_locations`, which splits `brief.location`
on `/`, `,`, `or`, `and` into individual cities, then passes if *any* one of them
matches the candidate's location by the existing per-city substring rule.
**Rejected** changing `ProjectBrief.location` to a `list[str]` in the schema --
this is a deterministic-Python filter bug (CLAUDE.md rule 1 territory), not an
extraction-prompt ambiguity, so per the project's convention (see phase 2) the
fix stays in `src/match.py`'s filter logic rather than growing the schema.

**Assumptions made:** the same per-city substring rule (either direction) still
applies per split city -- e.g. a brief city of "Denmark" still matches any
Danish city, and a candidate's fuller "Aarhus, Denmark, Europe" still matches a
brief's plain "Aarhus". Only the multi-city splitting is new.

**Known limitations:** the split regex handles the delimiters actually observed
from the LLM's extraction (`/`, `,`, `or`, `and`) -- a brief location string using
some other separator (e.g. `;` or `&`) would still be treated as one un-split
city and could silently under-match. Not fixed here since no such case has been
observed; would need a real example to design against, per this project's
convention of not guessing at unvalidated edge cases.

## Post-phase-4 UI tweak — moved "Data quality" into General info

**What was built:** In `app.py`, moved the "Data quality" panel out of section
4's per-run tabs (alongside the Earliest start / Lowest cost alternative teams)
and into the "General info" section at the top of the page, as a third expander
next to "Team availability" and "Consultant network (team chemistry)".

**Key design decision:** `render_data_quality` only ever reads `profiles` and
`names_by_id` -- both loaded once at app startup -- and never touches
`brief`/`result` from the current run. Living inside the per-run tabs implied it
was a property of that run's matching result, which it isn't: it reports the
same trust-flag and extraction-note findings for the whole corpus regardless of
which brief was just typed in. General info is where every other run-independent
panel already lives, so this is a placement fix, not a behavioural change --
`render_data_quality` itself is untouched.

**Assumptions made:** none new; section 4 was renamed from "Alternatives & data
quality" back to plain "Alternatives" now that it only holds the two alternative
teams.
