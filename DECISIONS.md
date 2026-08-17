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
