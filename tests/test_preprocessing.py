import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_text, select_train_dev


def test_keeps_accents_negations_and_punctuation():
    t = "No me atendieron, ¡sin ninguna explicación!"
    assert clean_text(t) == t


def test_collapses_whitespace_and_strips():
    assert clean_text("  hola \n  amigo\t ") == "hola amigo"


def test_unicode_nfc_equivalence():
    assert clean_text("cafe\u0301") == clean_text("café")


def test_idempotent():
    s = "  ¿Cuándo es   mi cita? "
    assert clean_text(clean_text(s)) == clean_text(s)


def test_select_train_dev_excludes_test():
    df = pd.DataFrame({"split": ["train", "dev", "test", "test"], "x": range(4)})
    out = select_train_dev(df)
    assert set(out["split"]) == {"train", "dev"} and len(out) == 2
