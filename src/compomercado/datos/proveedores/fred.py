"""Series macro y de estrés desde FRED (CSV público, sin API key)."""

from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"
UA = {"User-Agent": "Mozilla/5.0 (compomercado; +https://github.com/jcruzburgos04-ops/Compomercado)"}


def parsear_csv(texto: str, serie: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(texto))
    if df.shape[1] < 2:
        raise ValueError(f"FRED {serie}: formato inesperado")
    fechas = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    col = serie if serie in df.columns else df.columns[1]
    valores = pd.to_numeric(df[col], errors="coerce")
    s = pd.Series(valores.to_numpy(), index=fechas, name=serie)
    return s[s.index.notna()].dropna().sort_index()


def descargar_serie(serie: str, reintentos: int = 3, pausa: float = 2.0) -> pd.Series:
    ultimo: Exception | None = None
    for intento in range(reintentos):
        try:
            r = requests.get(URL.format(serie=serie), headers=UA, timeout=60)
            r.raise_for_status()
            return parsear_csv(r.text, serie)
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(pausa * (2**intento))
    raise RuntimeError(f"FRED {serie}: {ultimo}")


def descargar(series: list[str]) -> tuple[pd.DataFrame, list[str]]:
    datos, fallidas = {}, []
    for s in series:
        try:
            datos[s] = descargar_serie(s)
        except Exception as e:  # noqa: BLE001
            log.warning("%s", e)
            fallidas.append(s)
    tabla = pd.DataFrame(datos).sort_index() if datos else pd.DataFrame()
    return tabla, fallidas
