from pathlib import Path

import pytest

from src.ingest import ingest_directory

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def records():
    return ingest_directory(DATA_DIR)


def test_covers_every_file_in_data_raw(records):
    seen_files = {Path(r["source_file"]).name.split("#")[0] for r in records}
    expected_files = {p.name for p in DATA_DIR.iterdir() if p.is_file()}
    assert seen_files == expected_files


def test_produces_21_records(records):
    # 14 pdf + 5 docx + 2 pptx slides (one CV per slide) = 21
    assert len(records) == 21


def test_every_record_has_substantial_text(records):
    for r in records:
        assert len(r["raw_text"]) >= 200, (
            f"{r['source_file']} produced only {len(r['raw_text'])} chars"
        )


def test_record_shape(records):
    for r in records:
        assert set(r.keys()) == {"source_file", "format", "raw_text", "extracted_at"}
        assert r["format"] in {"pdf", "docx", "pptx"}
        assert isinstance(r["raw_text"], str)


def test_danish_letters_in_pdf_names_are_not_corrupted(records):
    # These PDFs render "ø" as a Type3 glyph with no ToUnicode CMap; a naive
    # extraction silently mis-maps it to "l with stroke" (ł) via Adobe
    # StandardEncoding. Confirm the fallback in src/ingest.py resolves it
    # to the real Danish letter instead.
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    assert "Bøgh" in by_source["cv_10_Victor_Bøgh_BI_Consultant.pdf"]
    assert "Bjørk" in by_source["cv_5_Emma_Bjørk_Analytics_Consultant.pdf"]
    assert "Mørk" in by_source["cv_9_Ida_Mørk_ML_Engineer.pdf"]
    for text in by_source.values():
        assert "ł" not in text and "Ł" not in text


def test_pdf_ligature_and_bullet_placeholders_are_resolved(records):
    # "fi"/"fl"/"ff"/"ffi" ligatures, an en dash, and a bullet marker all
    # come through pdfminer as bare "(cid:N)" placeholders in these PDFs
    # (no ToUnicode mapping at all). Confirm the six known codes are
    # substituted with their real characters rather than left as noise.
    by_source = {r["source_file"]: r["raw_text"] for r in records}
    assert "firm" in by_source["cv_10_Victor_Bøgh_BI_Consultant.pdf"]
    assert "efforts" in by_source["cv_9_Ida_Mørk_ML_Engineer.pdf"]
    assert "fluent" in by_source["cv_12_Christian_Enevoldsen_Cloud_Architect.pdf"]
    assert "efficiency" in by_source["cv_11_Anna_Skov_Digital_Transformation_Consultant.pdf"]
    assert "•" in by_source["cv_10_Victor_Bøgh_BI_Consultant.pdf"]
    for text in by_source.values():
        for code in ("21", "27", "28", "29", "30", "136"):
            assert f"(cid:{code})" not in text


def test_docx_table_cv_has_text(records):
    # One CV stores everything in tables rather than paragraphs; make sure
    # the table-extraction path actually pulls that content out.
    docx_records = [r for r in records if r["format"] == "docx"]
    assert docx_records, "expected at least one docx record"
    assert all(len(r["raw_text"]) >= 200 for r in docx_records)


def test_pptx_splits_into_one_record_per_slide(records):
    pptx_records = [r for r in records if r["format"] == "pptx"]
    assert len(pptx_records) == 2
    sources = {r["source_file"] for r in pptx_records}
    assert sources == {
        "cvs-ppt-format.pptx#slide1",
        "cvs-ppt-format.pptx#slide2",
    }


def test_bad_file_is_skipped_not_raised(tmp_path):
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf")
    (tmp_path / "good.docx").write_bytes(b"also not a real docx")
    # Should not raise even though both files are invalid.
    result = ingest_directory(tmp_path)
    assert result == []


def test_unsupported_extensions_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("irrelevant")
    result = ingest_directory(tmp_path)
    assert result == []


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_directory(tmp_path / "does_not_exist")
