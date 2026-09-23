"""Series de precio de canastas propias (config/canastas.yaml)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Canasta

log = logging.getLogger(__name__)

FRECUENCIAS = {"diario": "D", "mensual": "ME", "trimestral": "QE"}


def pesos_en(fecha, canasta: Canasta, r: pd.DataFrame, disponibles: list[str]) -> pd.Series:
    if canasta.ponderacion == "manual":
        w = pd.Series({t: p for t, p in canasta.componentes if t in disponibles and p is not None}, dtype=float)
    elif canasta.ponderacion == "inversa_volatilidad":
        vol = r.loc[:fecha, disponibles].tail(63).std()
        w = (1 / vol.replace(0, np.nan)).dropna()
    else:
        # "igual" y "capitalizacion" (sin historia gratuita de capitalización: se usa igual peso).
        w = pd.Series(1.0, index=disponibles)
    return w / w.sum() if w.sum() > 0 else w


def serie_canasta(precios: pd.DataFrame, canasta: Canasta) -> pd.Series:
    """Índice base 100 de la canasta, con pesos fijos entre rebalanceos.

    En cada día, solo cuentan los componentes con precio (los pesos se renormalizan).
    """
    tickers = [t for t in canasta.tickers if t in precios.columns]
    if not tickers:
        return pd.Series(dtype=float, name=canasta.nombre)
    p = precios[tickers]
    r = p / p.shift(1) - 1
    frecuencia = FRECUENCIAS.get(canasta.rebalanceo, "ME")
    fechas_reb = set(p.resample(frecuencia).last().index) if frecuencia != "D" else set(p.index)
    w = pd.Series(dtype=float)
    retornos = []
    for fecha in p.index:
        disponibles = [t for t in tickers if pd.notna(p.at[fecha, t])]
        if w.empty or fecha in fechas_reb or not set(w.index) <= set(disponibles):
            w = pesos_en(fecha, canasta, r, disponibles) if disponibles else pd.Series(dtype=float)
        ri = r.loc[fecha, w.index].dropna() if not w.empty else pd.Series(dtype=float)
        wi = w.reindex(ri.index)
        retornos.append(float((ri * wi).sum() / wi.sum()) if wi.sum() > 0 else np.nan)
    serie = pd.Series(retornos, index=p.index).fillna(0)
    inicio = p.dropna(how="all").index.min()
    serie = serie.loc[inicio:]
    return (100 * (1 + serie).cumprod()).rename(canasta.nombre)
