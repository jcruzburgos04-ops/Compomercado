import numpy as np
import pandas as pd

from compomercado.eventos import caidas


def serie(valores):
    return pd.Series(valores, index=pd.bdate_range("2020-01-01", periods=len(valores)), dtype=float)


def test_zigzag_detecta_tramos_y_rebote():
    p = serie([100, 105, 110, 99, 95, 105, 106, 120, 100, 96, 97])
    eps = caidas.tramos_zigzag(p, umbral=0.10)
    assert len(eps) == 2
    primero, segundo = eps.iloc[0], eps.iloc[1]
    assert primero["precio_pico"] == 110 and primero["precio_valle"] == 95
    assert primero["confirmacion"] == p.index[3]  # 99 <= 110 * 0.9
    assert primero["fin"] == p.index[5]  # 105 >= 95 * 1.1
    assert not primero["en_curso"]
    assert primero["recuperacion"] == p.index[7]
    assert segundo["precio_pico"] == 120 and segundo["precio_valle"] == 96
    assert segundo["en_curso"] and pd.isna(segundo["fin"])
    assert np.isclose(segundo["profundidad"], 96 / 120 - 1)


def test_zigzag_ignora_caidas_menores_al_umbral():
    p = serie([100, 95, 100, 96, 101, 97, 102])
    assert caidas.tramos_zigzag(p, umbral=0.10).empty


def test_bajo_agua_un_episodio_hasta_recuperar():
    p = serie([100, 110, 99, 95, 105, 98, 111, 112])
    eps = caidas.episodios_bajo_agua(p, umbral=0.10)
    assert len(eps) == 1
    e = eps.iloc[0]
    assert e["pico"] == p.index[1] and e["valle"] == p.index[3]
    assert e["recuperacion"] == p.index[6]
    assert e["dias_caida"] == 2


def test_bajo_agua_descarta_poco_profundos():
    p = serie([100, 110, 104, 111])
    assert caidas.episodios_bajo_agua(p, umbral=0.10).empty


def test_clasificar_por_reglas():
    eps = pd.DataFrame({"d_baa10y": [0.8, 0.0, 0.0], "d_dgs10": [0.0, 0.5, -0.1],
                        "dias_caida": [100, 60, 10], "vix_max": [40, 25, 20]})
    reglas = [
        {"tipo": "Credito", "condicion": "d_baa10y >= 0.50"},
        {"tipo": "Tasas", "condicion": "d_dgs10 >= 0.30"},
        {"tipo": "Otro", "condicion": "True"},
    ]
    assert caidas.clasificar(eps, reglas).tolist() == ["Credito", "Tasas", "Otro"]


def test_huella_macro(mercado):
    eps = caidas.tramos_zigzag(mercado, umbral=0.10)
    tasa = pd.Series(np.linspace(1, 3, len(mercado)), index=mercado.index)
    h = caidas.huella_macro(eps, {"dgs10": tasa}, {"usd": mercado}, {"vix": mercado})
    assert (h["d_dgs10"] > 0).all()
    assert np.allclose(h["r_usd"], eps["profundidad"])
