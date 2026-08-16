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
  could still be picked up as the "name" incorrectly.
- The stage-2 LLM call has no retry/backoff logic; a transient API failure
  during `scan_for_injection` currently propagates as an exception rather
  than being retried or degraded to a lower-confidence heuristic-only
  result.
