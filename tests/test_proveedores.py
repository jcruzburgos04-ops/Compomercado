import pandas as pd

from compomercado.analitica import canastas
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


def test_resumir_cadena_rechaza_spot_invalido():
    import pytest

    from compomercado.datos import opciones

    cadena = pd.DataFrame({"tipo": ["call", "put"], "vencimiento": ["2026-10-16"] * 2, "strike": [100.0, 95.0],
                           "volume": [10, 20], "openInterest": [100, 300], "impliedVolatility": [0.2, 0.25]})
    with pytest.raises(ValueError):
        opciones.resumir_cadena(cadena, float("nan"), pd.Timestamp("2026-09-22"))
    r = opciones.resumir_cadena(cadena, 100.0, pd.Timestamp("2026-09-22"))
    assert r["pc_oi"] == 3.0 and abs(r["iv_atm_30"] - 0.2) < 1e-12 and abs(r["skew_95_30"] - 0.05) < 1e-12


def test_opciones_validas_ignora_snapshots_sin_spot():
    import numpy as np

    from compomercado.pipeline import _opciones_validas

    o = pd.DataFrame({"SPY_spot": [np.nan, 650.0], "SPY_iv_atm_30": [1.74, 0.15], "QQQ_spot": [580.0, 585.0],
                      "QQQ_iv_atm_30": [0.2, 0.21]}, index=pd.to_datetime(["2026-09-22", "2026-09-23"]))
    v = _opciones_validas(o)
    assert np.isnan(v.loc["2026-09-22", "SPY_iv_atm_30"]) and v.loc["2026-09-22", "QQQ_iv_atm_30"] == 0.2


def _xlsx_ssga() -> bytes:
    import io

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Fund Name:", "SPDR S&P 500 ETF Trust"])
    ws.append(["Ticker Symbol:", "SPY"])
    ws.append(["Holdings:", "As of 22-Sep-2026"])
    ws.append([])
    ws.append(["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"])
    ws.append(["NVIDIA CORP", "NVDA", "x", "x", 7.5, "Information Technology", 1, "USD"])
    ws.append(["BERKSHIRE HATHAWAY INC CL B", "BRK.B", "x", "x", 1.6, "Financials", 1, "USD"])
    ws.append(["US DOLLAR", "CASH_USD", "x", "x", 0.1, "-", 1, "USD"])
    ws.append(["EMPRESA ESCINDIDA", "2602335D", "x", "x", 0.01, "-", 1, "USD"])
    ws.append([])
    ws.append(["Past performance is no guarantee of future results."])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_parser_ssga_spy():
    from compomercado.datos.proveedores import sp500

    t = sp500.parsear_ssga(_xlsx_ssga())
    assert list(t["ticker"]) == ["NVDA", "BRK-B"]
    assert abs(t["peso"].iloc[0] - 0.075) < 1e-12
    assert t["fecha"].iloc[0] == pd.Timestamp("2026-09-22")


def test_parser_wikipedia_sp500():
    from compomercado.datos.proveedores import sp500

    html = ("<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>"
            "<tr><td>BF.B</td><td>Brown-Forman</td><td>Consumer Staples</td></tr>"
            "<tr><td>AAPL</td><td>Apple</td><td>Information Technology</td></tr></table>")
    t = sp500.parsear_wikipedia(html)
    assert list(t["ticker"]) == ["BF-B", "AAPL"] and t["peso"].isna().all()


def test_parser_fomc():
    from compomercado.datos.proveedores import calendario as cal

    html = """<h4><a>2026 FOMC Meetings</a></h4>
    <div class="fomc-meeting__month"><strong>January</strong></div><div class="fomc-meeting__date">27-28</div>
    <p>Minutes: Released February 18, 2026</p>
    <div class="fomc-meeting__month"><strong>April/May</strong></div><div class="fomc-meeting__date">28-1*</div>
    <div class="fomc-meeting__month"><strong>December</strong></div><div class="fomc-meeting__date">8-9*</div>
    <h4><a>2025 FOMC Meetings</a></h4>
    <div class="fomc-meeting__month"><strong>September</strong></div><div class="fomc-meeting__date">16-17*</div>"""
    t = cal.parsear_fomc(html)
    assert list(t["fecha"]) == [pd.Timestamp("2025-09-17"), pd.Timestamp("2026-01-28"),
                                pd.Timestamp("2026-05-01"), pd.Timestamp("2026-12-09")]


def test_parser_bls_solo_tablas():
    from compomercado.datos.proveedores import calendario as cal

    html = """<p>Last Modified Date: Jan 10, 2026</p><table><tr><th>Reference Month</th><th>Release Date</th></tr>
    <tr><td>September 2026</td><td>Oct. 15, 2026</td><td>08:30 AM</td></tr>
    <tr><td>October 2026</td><td>Nov. 12, 2026</td><td>08:30 AM</td></tr></table>"""
    t = cal.parsear_bls(html, "cpi")
    assert list(t["fecha"]) == [pd.Timestamp("2026-10-15"), pd.Timestamp("2026-11-12")]


def test_screener_consultas_y_condiciones():
    from compomercado.analitica import screener

    t = pd.DataFrame({"grupo": ["sectores", "acciones", "acciones"], "rs_63_pct": [90, 75, 20],
                      "beta_bajista": [0.8, 1.5, 1.4]}, index=["XLP", "AAA", "BBB"])
    screens = {
        "fuertes": {"consulta": "rs_63_pct >= 70", "orden": "rs_63_pct"},
        "acciones_beta": {"grupos": ["acciones"], "consulta": "beta_bajista >= 1.2", "orden": "beta_bajista"},
        "solo_en_caida": {"consulta": "rs_63_pct >= 80", "solo_si": "drawdown_spy <= -0.05"},
        "roto": {"consulta": "columna_que_no_existe > 1"},
    }
    r = {s["clave"]: s for s in screener.aplicar(t, screens, {"drawdown_spy": -0.02})}
    assert list(r["fuertes"]["resultado"].index) == ["XLP", "AAA"]
    assert list(r["acciones_beta"]["resultado"].index) == ["AAA", "BBB"]
    assert r["solo_en_caida"]["estado"] == "no_aplica"
    assert r["roto"]["estado"] == "error"
