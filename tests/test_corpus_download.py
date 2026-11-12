import pytest

from src.corpus_sources import (
    AMERICASNLP_COMMIT,
    CORPUS_FILES,
    REQUIRED_CORPUS_FILES,
    RLTRANSLATOR_COMMIT,
)
from scripts.descargar_corpus import verify_content


def test_corpus_urls_and_checksums_are_pinned():
    assert len(AMERICASNLP_COMMIT) == 40
    assert set(REQUIRED_CORPUS_FILES) <= set(CORPUS_FILES)
    for source in CORPUS_FILES.values():
        assert len(source.sha256) == 64
        assert AMERICASNLP_COMMIT in source.url or RLTRANSLATOR_COMMIT in source.url


def test_checksum_mismatch_is_rejected():
    with pytest.raises(ValueError, match="SHA-256 incorrecto"):
        verify_content("train.es", b"contenido modificado")
