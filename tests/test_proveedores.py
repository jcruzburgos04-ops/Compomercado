import pandas as pd

from compomercado.analitica import argentina, canastas
from compomercado.config import Canasta
from compomercado.datos.proveedores import cboe, fred, french, yahoo
from compomercado.indicadores import estado


def test_parser_fred():
    texto = "observation_date,BAA10Y\n2024-01-02,1.70\n2024-01-03,.\n2024-01-04,1.72\n"
    s = fred.parsear_csv(texto, "BAA10Y")
    assert list(s.values) == [1.70, 1.72]
    assert s.index[0] == pd.Timestamp("2024-01-02")


def test_parser_cboe():
    texto = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n01/03/1990,18.19,18.19,18.19,18.19\n"
    s = cboe.parsear_csv(texto, "VIX")
    assert s.iloc[-1] == 18.19 and s.index[0] == pd.Timestamp("1990-01-02")
    texto2 = "DATE,VVIX\n2007-01-03,87.63\n"
    assert cboe.parsear_csv(texto2, "VVIX").iloc[0] == 87.63


def test_parser_french():
    texto = (
        "This file was created by CMPT_IND_RETS_DAILY using the 202607 CRSP database.\n"
        "It contains value- and equal-weighted returns for 49 industry portfolios.\n\n"
        "  Average Value Weighted Returns -- Daily\n"
        ",Agric,Food ,Soda \n"
        "19260701,   0.56,  -0.07, -99.99\n"
        "19260702,  -1.00,   0.50,   0.10\n"
        "\n"
        "  Average Equal Weighted Returns -- Daily\n"
        ",Agric,Food ,Soda \n"
        "19260701,   9.99,   9.99,   9.99\n"
    )
    df = french.parsear_bloque(texto, r"Value Weighted Returns")
    assert list(df.columns) == ["Agric", "Food", "Soda"]
    assert len(df) == 2
    assert abs(df.loc["1926-07-01", "Agric"] - 0.0056) < 1e-12
    assert pd.isna(df.loc["1926-07-01", "Soda"])


def test_quitar_barra_incompleta():
    from datetime import datetime

    idx = pd.to_datetime(["2026-09-21", "2026-09-22"])
    df = pd.DataFrame({"SPY": [1.0, 2.0]}, index=idx)
    durante = datetime(2026, 9, 22, 11, 0, tzinfo=yahoo.NY)
    despues = datetime(2026, 9, 22, 18, 0, tzinfo=yahoo.NY)
    assert len(yahoo.quitar_barra_incompleta(df, durante)) == 1
    assert len(yahoo.quitar_barra_incompleta(df, despues)) == 2


def test_ccl_descarta_especie_desalineada():
    idx = pd.bdate_range("2024-01-01", periods=3)
    locales = pd.DataFrame({"A.BA": [10000.0] * 3, "B.BA": [1000.0] * 3, "C.BA": [5000.0] * 3}, index=idx)
    adrs = pd.DataFrame({"A": [10.0] * 3, "B": [1.0] * 3, "C": [2.0] * 3}, index=idx)
    pares = [{"adr": "A", "local": "A.BA", "ratio": 1}, {"adr": "B", "local": "B.BA", "ratio": 1},
             {"adr": "C", "local": "C.BA", "ratio": 1}]  # C implica 2500: ratio mal cargado
    ccl, _ = argentina.ccl_implicito(locales, adrs, pares)
    assert (ccl == 1000).all()


def test_canasta_igual_peso():
    idx = pd.bdate_range("2024-01-01", periods=4)
    precios = pd.DataFrame({"A": [100, 110, 110, 110.0], "B": [100, 100, 90, 90.0]}, index=idx)
    c = Canasta("x", "x", "igual", "diario", (("A", None), ("B", None)))
    s = canastas.serie_canasta(precios, c)
    assert abs(s.iloc[1] - 105) < 1e-9
    assert abs(s.iloc[2] - 105 * (1 + 0.5 * -0.1)) < 1e-9


def test_caida_maxima_futura():
    p = pd.Series([100, 90, 95, 80, 100.0])
    f = estado.caida_maxima_futura(p, 2)
    assert abs(f.iloc[0] + 0.10) < 1e-12 and abs(f.iloc[1] - (80 / 90 - 1)) < 1e-12 and pd.isna(f.iloc[3])


def test_cta_estimado():
    p = pd.Series(range(1, 300), dtype=float)
    pos, _ = estado.cta_estimado(p)
    assert pos.iloc[-1] == 1.0


def test_completar_ajustado_ultima_rueda():
    idx = pd.bdate_range("2024-01-01", periods=3)
    tablas = {
        "cierre": pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=idx),
        "cierre_aj": pd.DataFrame({"SPY": [99.0, 100.0, float("nan")]}, index=idx),
    }
    yahoo.completar_ajustado(tablas)
    assert tablas["cierre_aj"]["SPY"].iloc[-1] == 102.0


def test_agregar_recientes_superpone_ultima_rueda(monkeypatch):
    idx_largo = pd.bdate_range("2024-01-01", periods=3)
    idx_corto = pd.bdate_range("2024-01-02", periods=3)
    tablas = {"cierre": pd.DataFrame({"SPY": [1.0, 2.0, 3.0]}, index=idx_largo)}
    corto = pd.DataFrame({("Close", "SPY"): [2.5, 3.5, 4.5]}, index=idx_corto)
    corto.columns = pd.MultiIndex.from_tuples(corto.columns)
    monkeypatch.setattr(yahoo, "_descargar_lote", lambda *a, **k: corto)
    monkeypatch.setattr(yahoo, "quitar_barra_incompleta", lambda df, ahora=None: df)
    yahoo._agregar_recientes(tablas, ["SPY"], 40, 1, 0)
    s = tablas["cierre"]["SPY"]
    assert list(s.values) == [1.0, 2.5, 3.5, 4.5]
