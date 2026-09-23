import numpy as np
import pandas as pd

from compomercado.indicadores import ponderacion as pond
from compomercado.indicadores.estado import Indicador


def test_auc_y_precision():
    idx = pd.RangeIndex(200)
    y = pd.Series([0] * 150 + [1] * 50, index=idx, dtype=float)
    perfecto = y * 10 + pd.Series(np.linspace(0, 1, 200), index=idx)
    assert pond.auc(perfecto, y) == 1.0
    assert pond.auc(-perfecto, y) == 0.0
    assert abs(pond.auc(pd.Series(1.0, index=idx), y) - 0.5) < 1e-12
    assert pond.precision_media(perfecto, y) == 1.0


def test_pesos_por_ranking_dan_mas_peso_al_mas_importante():
    w = pond.pesos_por_ranking(pd.Series({"a": 0.60, "b": 0.70, "c": 0.55}))
    assert list(w.index) == ["b", "a", "c"]
    assert np.allclose(w.values, [3 / 6, 2 / 6, 1 / 6])


def _escenario(n=4000, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2000-01-03", periods=n)
    y = pd.Series((rng.random(n) < 0.2).astype(float), index=idx)
    Y = pd.DataFrame({"y1": y, "y2": y, "y3": y})
    senal = y * 25 + rng.normal(50, 15, n)
    senal_debil = y * 8 + rng.normal(50, 15, n)
    R = pd.DataFrame({
        "bueno": senal,
        "copia": senal + rng.normal(0, 1, n),        # redundante con "bueno"
        "debil": senal_debil,                          # informa menos
        "ruido": rng.normal(50, 15, n),                # no informa
        "reves": 100 - senal,                          # funciona al revés
        "cred": y * 20 + rng.normal(50, 15, n),
        "tend": y * 15 + rng.normal(50, 15, n),
        "corto": pd.Series(senal).where(pd.Series(np.arange(n), index=idx) > n - 300),  # poca historia
    }, index=idx)
    pilares = {"bueno": "volatilidad", "copia": "volatilidad", "debil": "volatilidad", "ruido": "volatilidad",
               "reves": "volatilidad", "corto": "volatilidad", "cred": "credito", "tend": "tendencia"}
    inds = [Indicador(i, i, p, 1, R[i]) for i, p in pilares.items()]
    return inds, R, Y


def test_depuracion_y_pesos_dentro_del_pilar():
    inds, R, Y = _escenario()
    p = pond.calcular_pesos(inds, R, Y)
    t = p.indicadores
    assert t.loc["bueno", "estado"] == "incluido"
    assert t.loc["copia", "estado"] == "repite a bueno"
    assert t.loc["ruido", "estado"] == "no anticipa caídas"
    assert t.loc["reves", "estado"] == "funciona al revés de lo esperado"
    assert t.loc["corto", "estado"] == "poca historia"
    # El más importante pesa más: bueno (2/3) > debil (1/3).
    w = p.peso_ind["volatilidad"]
    assert w["bueno"] > w["debil"] > 0 and abs(w.sum() - 1) < 1e-12
    # Entre pilares también manda el ranking: volatilidad > crédito > tendencia.
    wp = p.peso_pilar
    assert wp["volatilidad"] > wp["credito"] > wp["tendencia"]


def test_compuesto_renormaliza_y_respeta_pesos():
    inds, R, Y = _escenario()
    p = pond.calcular_pesos(inds, R, Y)
    c = pond.compuesto(R, p)
    assert c["total"].dropna().between(0, 100).all()
    w = p.peso_ind["volatilidad"]
    esperado = (R["bueno"] * w["bueno"] + R["debil"] * w["debil"]) / (w["bueno"] + w["debil"])
    assert np.allclose(c["volatilidad"], esperado)


def test_pesos_no_miran_el_futuro():
    inds, R, Y = _escenario()
    corte = R.index[2500]
    p1 = pond.calcular_pesos(inds, R, Y, hasta=corte)
    R2, Y2 = R.copy(), Y.copy()
    R2.loc[R2.index > corte] = 0.0      # cambiar todo lo posterior al corte
    Y2.loc[Y2.index > corte] = 1.0
    p2 = pond.calcular_pesos(inds, R2, Y2, hasta=corte)
    pd.testing.assert_frame_equal(p1.indicadores, p2.indicadores)
    pd.testing.assert_series_equal(p1.peso_pilar, p2.peso_pilar)


def test_walk_forward_genera_serie_fuera_de_muestra():
    inds, R, Y = _escenario()
    pil, dentro, vigentes, historial = pond.walk_forward(inds, R, Y, primer_anio=2004)
    assert vigentes is not None and not pil.empty
    # Los pesos de cada año se estiman antes de que empiece ese año.
    assert vigentes.hasta < pd.Timestamp(f"{pil.index.max().year}-01-01")
    assert pil.index.min().year >= 2004
    assert set(historial.columns) >= {"volatilidad", "credito", "tendencia"}
