"""Genera un proyecto con datos sintéticos (sin red) para tests de punta a punta."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from compomercado.config import Proyecto
from compomercado.datos.almacen import Almacen
from compomercado.datos.calidad import reporte_precios
from compomercado.pipeline import INDUSTRIAS_FF

RAIZ_REPO = Path(__file__).resolve().parents[1]


def crear(tmp: Path, inicio="2012-01-02", fin="2026-09-18", semilla=0) -> Proyecto:
    shutil.copytree(RAIZ_REPO / "config", tmp / "config")
    p = Proyecto(tmp)
    alm = Almacen(p.dir_datos)
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range(inicio, fin)
    n = len(fechas)

    # Mercado con dos caídas marcadas.
    rm = rng.normal(0.0004, 0.009, n)
    rm[800:860] -= 0.006
    rm[2000:2040] -= 0.008
    tickers = p.tickers_yahoo()
    cierre = {}
    for k, t in enumerate(tickers):
        beta = 0.3 + 1.4 * ((k * 37) % 100) / 100
        r = beta * rm + rng.normal(0, 0.006, n)
        precio = 50 * np.cumprod(1 + r)
        if t.startswith("^VIX") or t in ("^VVIX", "^MOVE"):
            precio = 15 + 10 * np.abs(pd.Series(rm).rolling(21, min_periods=1).std().to_numpy()) / 0.009
        cierre[t] = precio
    cierre = pd.DataFrame(cierre, index=fechas)
    cierre["SPY"] = 100 * np.cumprod(1 + rm)
    cierre.iloc[:300, cierre.columns.get_loc("XLC")] = np.nan  # activo que empieza después
    alm.guardar("precios/cierre_aj", cierre)
    alm.guardar("precios/cierre", cierre)
    alm.guardar("precios/apertura", cierre * 0.999)
    alm.guardar("precios/maximo", cierre * 1.005)
    alm.guardar("precios/minimo", cierre * 0.995)
    alm.guardar("precios/volumen", pd.DataFrame(rng.integers(1e6, 5e6, size=cierre.shape), index=fechas, columns=cierre.columns).astype(float))

    fred = pd.DataFrame({s: 2 + np.cumsum(rng.normal(0, 0.02, n)) for s in p.series_fred}, index=fechas)
    alm.guardar("fred", fred)
    cboe = pd.DataFrame({s: cierre.get(f"^{s}", pd.Series(20.0, index=fechas)) for s in p.series_cboe}, index=fechas)
    alm.guardar("cboe", cboe)

    fechas_ff = pd.bdate_range("1990-01-02", "2026-06-30")
    rff = rng.normal(0.0003, 0.01, len(fechas_ff))
    rff[1000:1100] -= 0.006
    rff[5000:5080] -= 0.007
    ind = pd.DataFrame({c: 1.0 * rff * (0.5 + (k % 10) / 10) + rng.normal(0, 0.005, len(fechas_ff))
                        for k, c in enumerate(INDUSTRIAS_FF)}, index=fechas_ff)
    alm.guardar("french/49_Industry_Portfolios_daily", ind)
    alm.guardar("french/F-F_Research_Data_Factors_daily",
                pd.DataFrame({"Mkt-RF": rff, "SMB": 0.0, "HML": 0.0, "RF": 0.0001}, index=fechas_ff))
    for fuente in ("yahoo", "fred", "cboe", "ken_french"):
        alm.registrar_descarga(fuente, origen="sintetico", fallidos=[])
    reporte_precios(cierre, pd.DatetimeIndex(fechas), set()).to_csv(p.dir_datos / "calidad_precios.csv")
    return p
