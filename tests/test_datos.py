import pandas as pd
import pytest

from src.config import CLASES, DATASET_PQRS, TEST_EXTERNO
from src.datos import cargar_todo, particion_por_plantilla
from src.generar_dataset import generar

COLUMNAS = ["id_msg", "texto", "tipo_pqrs", "intencion", "registro", "wayuunaiki", "plantilla_id", "fuente"]


@pytest.fixture(scope="module")
def D():
    return cargar_todo()


def test_esquema_dataset():
    d = pd.read_csv(DATASET_PQRS, keep_default_na=False)
    assert list(d.columns) == COLUMNAS
    assert set(d.tipo_pqrs) == set(CLASES)
    assert d.id_msg.is_unique
    assert set(d.registro) == {"formal", "coloquial", "simplificado"}


def test_generador_reproducible():
    """El CSV versionado debe coincidir con lo que produce el generador con la semilla por defecto."""
    nuevo = generar().astype(str)
    guardado = pd.read_csv(DATASET_PQRS, keep_default_na=False, dtype=str).head(len(nuevo))
    pd.testing.assert_frame_equal(nuevo.reset_index(drop=True), guardado.reset_index(drop=True))


def test_test_externo_valido():
    e = pd.read_csv(TEST_EXTERNO, keep_default_na=False)
    assert set(e.tipo_pqrs) <= set(CLASES) and len(e) >= 60
    assert not e.texto.isin(set(pd.read_csv(DATASET_PQRS, keep_default_na=False).texto)).any()


def test_sin_fuga_entre_particiones(D):
    tr, va, te = (set(D[s].plantilla_id) for s in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert not D["test"].texto_norm.isin(set(D["train"].texto_norm)).any()


def test_todas_las_clases_en_cada_particion(D):
    for s in ("train", "val", "test"):
        assert set(D[s].tipo_pqrs) == set(CLASES), s


def test_sin_duplicados_ni_vacios(D):
    d = D["d"]
    assert not d.duplicated(["texto_norm", "tipo_pqrs"]).any()
    assert (d.n_palabras > 0).all()
    assert (D["sin_contenido"].n_palabras == 0).all()


def test_particion_determinista(D):
    a = particion_por_plantilla(D["d"], seed=7)
    b = particion_por_plantilla(D["d"], seed=7)
    assert a.equals(b)
