import numpy as np
import pandas as pd

from compomercado.analitica import anomalias as an
from compomercado.datos.almacen import Almacen
from compomercado.datos.series import Series
from compomercado.eventos import caidas
from compomercado.indicadores import estado

from . import sintetico


def test_episodios_con_periodo_refractario():
    fechas = pd.bdate_range("2020-01-01", periods=200)
    m = pd.Series(0.0, index=fechas)
    m.iloc[[10, 11, 12, 30, 60, 100, 101]] = 1
    ep = an.episodios(m, refractario=21)
    # 10-12 y 30 son la misma racha (menos de 21 ruedas sin marcas); 60 y 100 empiezan episodios nuevos.
    assert list(ep) == [fechas[10], fechas[60], fechas[100]]


def _mercado_con_caidas(n=3000, semilla=3):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2005-01-03", periods=n)
    r = rng.normal(0.0005, 0.003, n)
    picos = list(range(400, n - 100, 180))
    for p in picos:
        r[p + 1: p + 15] -= 0.008      # ~10 % de caída después de cada pico
    return pd.Series(100 * np.cumprod(1 + r), index=fechas), picos


def test_evaluar_distingue_un_detector_que_anticipa():
    spy, _ = _mercado_con_caidas()
    rng = np.random.default_rng(0)
    eps = caidas.tramos_zigzag(spy, 0.05)
    bueno = pd.Series(0.0, index=spy.index)
    for p in spy.index.get_indexer(eps["pico"]):
        bueno.iloc[p - 5] = 1                                     # marca una semana antes de cada pico
    ruido = pd.Series((rng.random(len(spy)) < 0.01).astype(float), index=spy.index)
    ev = an.evaluar([an.Detector("bueno", "Bueno", "x", "", bueno), an.Detector("ruido", "Ruido", "x", "", ruido)],
                    spy, eps)
    assert ev.loc["bueno", "lift"] > 3 and ev.loc["bueno", "q"] < 0.01
    assert ev.loc["bueno", "cobertura_antes"] > 0.8 > ev.loc["bueno", "azar_antes"] * 3
    assert ev.loc["bueno", "veredicto"].startswith("Anticipa")
    assert ev.loc["ruido", "veredicto"] in {"No anticipa caídas", "Sugiere riesgo, sin significancia"}
    assert abs(ev.loc["ruido", "lift"] - 1) < 0.6


def test_efectos_de_calendario_detecta_el_cambio_de_mes_y_los_feriados():
    fechas = pd.bdate_range("1990-01-01", "2024-12-31")
    fechas = fechas[~fechas.isin(pd.to_datetime(["2020-07-03", "2019-12-25"]))]   # dos feriados
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, len(fechas)), index=fechas)
    cond = an._condiciones(fechas)
    r[cond["cambio_de_mes"][1]] += 0.002
    tabla = an.efectos_calendario(r)
    assert tabla.loc["cambio_de_mes", "t_Todo"] > 4 and tabla.loc["cambio_de_mes", "vigencia"] == "Sigue vigente"
    assert abs(tabla.loc["lunes", "t_Todo"]) < 3
    pre = pd.Series(cond["pre_feriado"][1], index=fechas)
    assert pre[pre].index.strftime("%Y-%m-%d").tolist() == ["2019-12-24", "2020-07-02"]
    # El último día de cada mes y los tres primeros.
    m = pd.Series(cond["cambio_de_mes"][1], index=fechas).loc["2024-03"]
    assert m[m].index.day.tolist() == [1, 4, 5, 29]


def test_detectores_no_miran_el_futuro(tmp_path):
    p = sintetico.crear(tmp_path / "completo")
    corte = pd.Timestamp("2021-06-15")
    q = sintetico.crear(tmp_path / "truncado")
    alm = Almacen(q.dir_datos)
    for nombre in ["precios/cierre_aj", "precios/cierre", "precios/apertura", "precios/maximo", "precios/minimo",
                   "precios/volumen", "cboe", "fred", "cftc/tff"]:
        alm.guardar(nombre, alm.leer(nombre).loc[:corte])

    def marcas(proy):
        S = Series(proy)
        return {d.id: d.serie for d in an.detectar(S, S.en_calendario(), estado.calcular(S))}

    completo, truncado = marcas(p), marcas(q)
    assert len(truncado) >= 5
    fecha = truncado["panico"].index.max()
    for id_, s in truncado.items():
        a = completo[id_].loc[:fecha]
        pd.testing.assert_series_equal(a, s.loc[:fecha], check_names=False, obj=id_)
