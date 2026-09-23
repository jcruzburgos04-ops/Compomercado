import numpy as np
import pandas as pd

from compomercado.analitica import huellas as hu


def _tramos(fechas, picos, dur=10):
    return pd.DataFrame({
        "pico": fechas[picos], "confirmacion": fechas[np.array(picos) + dur // 2],
        "valle": fechas[np.array(picos) + dur], "profundidad": -0.06, "tipo": "Otro",
    })


def test_instantaneas_y_trayectoria_se_alinean_con_el_ancla():
    fechas = pd.bdate_range("2000-01-03", periods=400)
    X = pd.DataFrame({"a": np.arange(400, dtype=float)}, index=fechas)
    eps = _tramos(fechas, [100, 200, 300])
    s = hu.instantaneas(eps, X)
    assert list(s["pre21"]["a"]) == [79, 179, 279]
    assert list(s["pre5"]["a"]) == [95, 195, 295]
    assert list(s["conf"]["a"]) == [105, 205, 305]
    assert list(s["valle"]["a"]) == [110, 210, 310]
    t = hu.trayectoria(eps, X, "pico", antes=5, despues=5)
    assert t.loc[0, "a"] == 200 and t.loc[-5, "a"] == 195
    # Fuera del rango del índice queda vacío, no se inventa.
    eps_borde = _tramos(fechas, [3])
    assert np.isnan(hu.instantaneas(eps_borde, X)["pre5"]["a"].iloc[0])


def test_benjamini_hochberg():
    q = hu.benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.2, np.nan]))
    assert np.allclose(q.iloc[:4], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
    assert np.isnan(q.iloc[4])


def test_comparar_distingue_senal_de_ruido():
    rng = np.random.default_rng(1)
    n = 6000
    fechas = pd.bdate_range("2000-01-03", periods=n)
    picos = list(range(700, n - 100, 130))
    X = pd.DataFrame({"senal": rng.uniform(0, 100, n), "ruido": rng.uniform(0, 100, n),
                      "piso": rng.uniform(0, 100, n)}, index=fechas)
    for p in picos:
        X.iloc[p - 25: p - 2, 0] = rng.uniform(75, 100, 23)   # alto antes de cada pico
        X.iloc[p + 9: p + 12, 2] = rng.uniform(85, 100, 3)    # alto en el valle
    eps = _tramos(fechas, picos)
    lec = hu.lectura(hu.comparar(hu.instantaneas(eps, X), X))
    assert lec.loc["senal", "papel"].startswith("Anticipa")
    assert lec.loc["piso", "papel"].startswith("Marca el piso")
    assert lec.loc["ruido", "papel"] == "Sin patrón claro"
    assert lec.loc["senal", "alta_pre5"] > 0.9 and lec.loc["senal", "lift_pre5"] > 2


def _escenario_analogos(n=2500, semilla=0):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2000-01-03", periods=n)
    F = pd.DataFrame(rng.uniform(0, 100, (n, 7)), index=fechas, columns=list("abcdefg"))
    F.iloc[rng.random((n, 7)) < 0.05] = np.nan
    y = pd.Series((F["a"] > 80).astype(float), index=fechas)   # el estado explica el evento
    return F, y


def test_analogos_no_miran_el_futuro():
    F, y = _escenario_analogos()
    m = 1800
    prob = hu.probabilidad_analogos(F, y, k=20, min_pasado=300)
    # Cambiar el resultado de los últimos 21 días (todavía desconocido en m-1) no altera nada hasta m-1.
    y2 = y.copy()
    y2.iloc[m - hu.PURGA: m] = 1 - y2.iloc[m - hu.PURGA: m]
    y2.iloc[m:] = np.nan
    prob2 = hu.probabilidad_analogos(F.iloc[:m], y2.iloc[:m], k=20, min_pasado=300, bloque=37)
    pd.testing.assert_series_equal(prob.iloc[:m], prob2, check_names=False)
    assert prob.notna().sum() > 1500
    # Aprende la relación: prob alta cuando "a" es alto.
    ok = prob.notna()
    assert prob[ok & (F["a"] > 90)].mean() > 0.45   # base: 20 %
    assert prob[ok & (F["a"] < 50)].mean() < 0.10


def test_analogos_de_hoy_son_de_episodios_distintos():
    F, _ = _escenario_analogos()
    a = hu.analogos_de(F, F.index[-1], n=5, separacion=63, excluir_recientes=126)
    assert len(a) == 5
    pos = sorted(F.index.get_indexer(a.index))
    assert all(b - x >= 63 for x, b in zip(pos, pos[1:], strict=False))
    assert max(pos) < len(F) - 126
    assert a["distancia"].is_monotonic_increasing


def test_desde_la_calma_separa_rachas():
    fechas = pd.bdate_range("2000-01-03", periods=1000)
    eps = _tramos(fechas, [100, 130, 300, 330, 600])   # dos rachas y un tramo aislado
    calma = hu.desde_la_calma(eps, fechas, ruedas=63)
    assert list(calma) == [True, False, True, False, True]
