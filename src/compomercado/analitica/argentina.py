"""Módulo Argentina: CCL implícito, Merval en dólares y cuánto del riesgo es global vs local."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ccl_implicito(locales: pd.DataFrame, adrs: pd.DataFrame, pares: list[dict], tolerancia: float = 0.15):
    """CCL por especie = precio local * acciones por ADR / precio del ADR; se combina con la mediana.

    Las especies que se alejan más de `tolerancia` de la mediana del día se descartan (ratio mal
    cargado, precio viejo, etc.). Devuelve (ccl, tabla por especie).
    """
    por_especie = {}
    for par in pares:
        loc, adr = par["local"], par["adr"]
        if loc in locales.columns and adr in adrs.columns:
            por_especie[adr] = locales[loc] * float(par["ratio"]) / adrs[adr]
    tabla = pd.DataFrame(por_especie).replace([np.inf, -np.inf], np.nan)
    if tabla.empty:
        return pd.Series(dtype=float, name="ccl"), tabla
    mediana = tabla.median(axis=1)
    desvio = (tabla.div(mediana, axis=0) - 1).abs()
    filtrada = tabla.where(desvio <= tolerancia)
    ccl = filtrada.median(axis=1).where(filtrada.notna().sum(axis=1) >= 2)
    return ccl.rename("ccl"), tabla


def exposicion_global(r_local: pd.Series, factores: pd.DataFrame, ventana: int = 252) -> pd.DataFrame:
    """Regresión móvil de los retornos argentinos contra factores globales.

    Devuelve el R² móvil (qué parte del movimiento explica el mundo) y el residuo acumulado de
    63 ruedas (movimiento "local/político" no explicado por los factores).
    """
    df = pd.concat([r_local.rename("y"), factores], axis=1).dropna()
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
    r2, resid = {}, {}
    for i in range(ventana, len(df) + 1):
        yi, Xi = y[i - ventana : i], X[i - ventana : i]
        coef, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        e = yi - Xi @ coef
        tot = ((yi - yi.mean()) ** 2).sum()
        fecha = df.index[i - 1]
        r2[fecha] = 1 - (e**2).sum() / tot if tot > 0 else np.nan
        resid[fecha] = e[-1]
    salida = pd.DataFrame({"r2_global": pd.Series(r2), "residuo": pd.Series(resid)})
    salida["residuo_63"] = salida["residuo"].rolling(63).sum()
    return salida
