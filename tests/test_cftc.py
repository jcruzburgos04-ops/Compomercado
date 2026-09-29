import numpy as np
import pandas as pd

from compomercado.datos.proveedores import cftc


def _fila(fecha, codigo, nombre, oi, **pos):
    return {"fecha": fecha, "codigo": codigo, "nombre": nombre, "oi": oi, **pos}


def test_normaliza_formato_api_y_archivo_anual():
    api = pd.DataFrame({
        "market_and_exchange_names": ["E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"],
        "report_date_as_yyyy_mm_dd": ["2024-01-02T00:00:00.000"],
        "cftc_contract_market_code": ["13874A"],
        "open_interest_all": ["1000"],
        "dealer_positions_long_all": ["100"], "dealer_positions_short_all": ["300"],
        "asset_mgr_positions_long": ["500"], "asset_mgr_positions_short": ["100"],
        "lev_money_positions_long": ["50"], "lev_money_positions_short": ["250"],
        "other_rept_positions_long": ["10"], "other_rept_positions_short": ["20"],
        "nonrept_positions_long_all": ["40"], "nonrept_positions_short_all": ["30"],
        "change_in_dealer_long_all": ["5"],
    })
    archivo = pd.DataFrame({
        "Market_and_Exchange_Names": ["E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"],
        "Report_Date_as_YYYY-MM-DD": ["2024-01-02"],
        "CFTC_Contract_Market_Code": ["13874A "],
        "Open_Interest_All": ["1000"],
        "Dealer_Positions_Long_All": ["100"], "Dealer_Positions_Short_All": ["300"],
        "Asset_Mgr_Positions_Long_All": ["500"], "Asset_Mgr_Positions_Short_All": ["100"],
        "Lev_Money_Positions_Long_All": ["50"], "Lev_Money_Positions_Short_All": ["250"],
        "Other_Rept_Positions_Long_All": ["10"], "Other_Rept_Positions_Short_All": ["20"],
        "NonRept_Positions_Long_All": ["40"], "NonRept_Positions_Short_All": ["30"],
    })
    a, b = cftc.normalizar(api), cftc.normalizar(archivo)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[0, "fecha"] == pd.Timestamp("2024-01-02") and a.loc[0, "codigo"] == "13874A"
    assert a.loc[0, "am_l"] == 500 and a.loc[0, "dealer_s"] == 300


def test_resolver_usa_el_contrato_con_mas_interes_abierto_y_el_nombre():
    base = {f"{g}_{s}": 1.0 for g in ("dealer", "am", "lev", "otros", "nr") for s in ("l", "s")}
    larga = pd.DataFrame([
        {"fecha": pd.Timestamp("2024-01-02"), "codigo": "13874A", "mercado_cftc": "E-MINI S&P 500 - CME", "oi": 100, **base},
        {"fecha": pd.Timestamp("2024-01-02"), "codigo": "13874+", "mercado_cftc": "S&P 500 CONSOLIDATED - CME", "oi": 150, **base},
        {"fecha": pd.Timestamp("2024-01-09"), "codigo": "13874A", "mercado_cftc": "E-MINI S&P 500 - CME", "oi": 110, **base},
        # Código desconocido pero nombre reconocible: entra por el patrón.
        {"fecha": pd.Timestamp("2024-01-02"), "codigo": "ZZZ999", "mercado_cftc": "RUSSELL E-MINI - CME", "oi": 70, **base},
        # Parecido pero no es el mercado (micro): no entra.
        {"fecha": pd.Timestamp("2024-01-02"), "codigo": "13874U", "mercado_cftc": "MICRO E-MINI S&P 500 INDEX - CME", "oi": 999, **base},
    ])
    ancha, usados = cftc.resolver(larga)
    assert list(ancha["sp500__oi"]) == [150, 110]
    assert ancha.loc["2024-01-02", "russell__oi"] == 70
    assert not any("MICRO" in u for u in usados["sp500"])


def test_neto_y_z_movil():
    t = pd.DataFrame({"oi": [100.0, 200.0], "am_l": [60.0, 50.0], "am_s": [20.0, 150.0]})
    assert np.allclose(cftc.neto(t, "am"), [0.4, -0.5])
    s = pd.Series(np.r_[np.zeros(60), 10.0])
    s.iloc[:60] = np.random.default_rng(0).normal(0, 1, 60)
    z = cftc.z_movil(s, semanas=156, minimo=52)
    assert z.iloc[:51].isna().all() and z.iloc[-1] > 3
    assert cftc.lectura(2.5) == "extremo comprado" and cftc.lectura(-1.2) == "más vendido que lo habitual"
