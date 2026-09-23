import numpy as np
import pandas as pd

from compomercado.analitica import comportamiento as comp
from compomercado.eventos import caidas


def precios_desde(r, indice):
    return pd.Series(100 * np.cumprod(1 + r), index=indice)


def test_beta_de_un_activo_apalancado(mercado):
    rm = comp.retornos(mercado)
    ra = 2 * rm
    m = comp.metricas_historicas(ra, rm)
    assert np.isclose(m["beta"], 2) and np.isclose(m["beta_bajista"], 2) and np.isclose(m["beta_alcista"], 2)
    assert np.isclose(m["corr"], 1)
    assert np.isclose(m["beta_cola"], 2)
    assert m["captura_bajista"] > 1.5


def test_activo_asimetrico_tiene_beta_bajista_menor(mercado):
    rm = comp.retornos(mercado)
    ra = np.where(rm < 0, 0.3 * rm, 1.0 * rm)
    ra = pd.Series(ra, index=rm.index)
    m = comp.metricas_historicas(ra, rm)
    assert np.isclose(m["beta_bajista"], 0.3, atol=0.01)
    assert np.isclose(m["beta_alcista"], 1.0, atol=0.01)
    assert m["asimetria"] < 0
    assert m["captura_bajista"] < 0.5


def test_metricas_por_episodio_defensivo_vs_ciclico(mercado):
    rm = comp.retornos(mercado).fillna(0)
    defensivo = precios_desde(0.5 * rm.to_numpy(), mercado.index)
    ciclico = precios_desde(1.5 * rm.to_numpy(), mercado.index)
    eps = caidas.tramos_zigzag(mercado, umbral=0.10)
    assert len(eps) >= 2
    d = comp.metricas_por_episodio(defensivo, mercado, eps)
    c = comp.metricas_por_episodio(ciclico, mercado, eps)
    assert (d["relativo"] > 0).all() and (c["relativo"] < 0).all()
    igual = comp.metricas_por_episodio(mercado * 3, mercado, eps)
    assert np.allclose(igual["lag_techo"], 0) and np.allclose(igual["lag_piso"], 0)
    assert np.allclose(igual["relativo"], 0)


def test_lag_de_techo_anticipado():
    idx = pd.bdate_range("2020-01-01", periods=200)
    m = pd.Series(np.r_[np.linspace(100, 120, 100), np.linspace(120, 90, 50), np.linspace(90, 110, 50)], index=idx)
    # El activo hace techo 20 ruedas antes que el mercado.
    a = pd.Series(np.r_[np.linspace(100, 130, 80), np.linspace(130, 80, 70), np.linspace(80, 100, 50)], index=idx)
    eps = caidas.tramos_zigzag(m, umbral=0.10)
    t = comp.metricas_por_episodio(a, m, eps)
    assert t["lag_techo"].iloc[0] == -20


def test_mapa_y_puntajes(mercado):
    rm = comp.retornos(mercado).fillna(0).to_numpy()
    rng = np.random.default_rng(1)
    precios = pd.DataFrame(
        {
            "MKT": mercado,
            "DEF": precios_desde(0.4 * rm + rng.normal(0, 0.003, len(rm)), mercado.index),
            "CIC": precios_desde(1.6 * rm + rng.normal(0, 0.003, len(rm)), mercado.index),
            "NEU": precios_desde(1.0 * rm + rng.normal(0, 0.003, len(rm)), mercado.index),
        }
    )
    eps = caidas.tramos_zigzag(mercado, umbral=0.05)
    tabla, larga = comp.mapa(precios, "MKT", eps)
    assert tabla.loc["DEF", "puntaje_refugio"] > tabla.loc["NEU", "puntaje_refugio"] > tabla.loc["CIC", "puntaje_refugio"]
    assert tabla.loc["CIC", "puntaje_rebote"] > tabla.loc["DEF", "puntaje_rebote"]
    assert set(larga["activo"]) == {"MKT", "DEF", "CIC", "NEU"}
    tipos = pd.Series(["A"] * len(eps), index=eps.index)
    pt = comp.por_tipo(larga, tipos)
    assert pt.loc["DEF"].iloc[0] > 0


def test_metricas_semanales_para_asincronicos(mercado):
    precios = pd.DataFrame({"MKT": mercado, "ASI": mercado.shift(1)})
    eps = caidas.tramos_zigzag(mercado, umbral=0.10)
    tabla, _ = comp.mapa(precios, "MKT", eps, asincronicos={"ASI"})
    assert tabla.loc["ASI", "frecuencia"] == "semanal"
    # Con retornos semanales, el desfase de un día casi no afecta la correlación.
    assert tabla.loc["ASI", "corr"] > 0.7
