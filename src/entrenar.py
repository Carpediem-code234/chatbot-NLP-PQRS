"""Entrena y evalúa los modelos de la Fase 2 y guarda el modelo de despliegue.

Uso:
    python -m src.entrenar               # evaluación completa + 10 particiones + modelo final
    python -m src.entrenar --rapido      # sin el experimento de 10 particiones

Salidas:
    results/resultados_modelos.csv       métricas por modelo y conjunto (val, test, externo)
    results/robustez_10_particiones.csv  F1-macro en 10 particiones por plantilla
    results/umbral_tau.csv               curva cobertura/accuracy del umbral en validación
    models/modelo_pqrs_m3.joblib         M3 reentrenado con todos los datos + umbral τ
"""
import argparse

import joblib
import pandas as pd

from src.config import MODELO, MODELS, RESULTS
from src.datos import cargar_todo
from src.evaluacion import curva_umbral, elegir_tau, evaluar, robustez
from src.modelos import MODELOS, m1_baseline, m2_nb, m3_propuesto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="omite el experimento de 10 particiones")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

    D = cargar_todo()
    train, val, test, ext, d = D["train"], D["val"], D["test"], D["externo"], D["d"]
    print(f"Mensajes sin contenido (regla R0): {len(D['sin_contenido'])} | limpio: {len(d):,} "
          f"| train {len(train)} · val {len(val)} · test {len(test)} · externo {len(ext)}")

    filas, modelos = [], {}
    for nombre, ctor in MODELOS.items():
        m = ctor().fit(train, train.tipo_pqrs)
        modelos[nombre] = m
        for conj, X in [("val", val), ("test", test), ("externo", ext)]:
            filas.append({"modelo": nombre, "conjunto": conj, **evaluar(m, X)})
    res = pd.DataFrame(filas)
    res.round(4).to_csv(RESULTS / "resultados_modelos.csv", index=False)
    print("\nF1-macro (partición principal):")
    print(res.pivot(index="modelo", columns="conjunto", values="F1_macro")[["val", "test", "externo"]].round(3).to_string())

    if not a.rapido:
        rob = robustez({"M1 Baseline": m1_baseline, "M2 ComplementNB": m2_nb, "M3 Propuesto": m3_propuesto}, d, ext)
        rob.to_csv(RESULTS / "robustez_10_particiones.csv", index=False)
        print("\nF1-macro en 10 particiones (media ± desv.):")
        print(rob.groupby("modelo")[["test (plantillas no vistas)", "test externo"]].agg(["mean", "std"]).round(3).to_string())

    curva = curva_umbral(modelos["M3 Propuesto"], val)
    curva.to_csv(RESULTS / "umbral_tau.csv", index=False)
    tau = elegir_tau(curva)
    print(f"\nUmbral τ elegido en validación: {tau}")

    final_tipo = m3_propuesto().fit(d, d.tipo_pqrs)
    final_int = m3_propuesto().fit(d, d.intencion)
    joblib.dump({"tipo": final_tipo, "intencion": final_int, "tau": tau}, MODELO)
    print(f"Modelo de despliegue guardado en {MODELO.relative_to(MODELO.parents[1])}")


if __name__ == "__main__":
    main()
