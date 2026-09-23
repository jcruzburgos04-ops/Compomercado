"""Historia diaria de la familia VIX desde el CDN público de CBOE."""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
UA = {"User-Agent": "Mozilla/5.0 (compomercado)"}


def parsear_csv(texto: str, sym: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(texto))
    df.columns = [str(c).strip().upper() for c in df.columns]
    col_fecha = "DATE" if "DATE" in df.columns else df.columns[0]
    fechas = pd.to_datetime(df[col_fecha], format="%m/%d/%Y", errors="coerce")
    if fechas.isna().all():
        fechas = pd.to_datetime(df[col_fecha], errors="coerce")
    if "CLOSE" in df.columns:
        col = "CLOSE"
    elif sym.upper() in df.columns:
        col = sym.upper()
    else:
        col = df.columns[-1]
    valores = pd.to_numeric(df[col], errors="coerce")
    s = pd.Series(valores.to_numpy(), index=fechas, name=sym)
    s = s[s.index.notna()].dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def descargar(simbolos: list[str]) -> tuple[pd.DataFrame, list[str]]:
    datos, fallidos = {}, []
    for sym in simbolos:
        try:
            r = requests.get(URL.format(sym=sym), headers=UA, timeout=60)
            r.raise_for_status()
            datos[sym] = parsear_csv(r.text, sym)
        except Exception as e:  # noqa: BLE001
            log.warning("CBOE %s: %s", sym, e)
            fallidos.append(sym)
    tabla = pd.DataFrame(datos).sort_index() if datos else pd.DataFrame()
    return tabla, fallidos
