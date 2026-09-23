"""Componentes actuales del S&P 500 y su peso.

Fuente principal: el archivo diario de tenencias de SPY que publica State Street (SSGA), con el
peso de cada empresa. Respaldo: la lista de Wikipedia (sin pesos).

Solo están las empresas que forman el índice hoy: calcular amplitud histórica con ellas tiene
sesgo de supervivencia (las que salieron del índice por caer no están).
"""

from __future__ import annotations

import io
import logging
from datetime import date

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

URL_SSGA = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/"
            "holdings-daily-us-en-spy.xlsx")
URL_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CABECERAS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) compomercado/0.1"}


def a_yahoo(ticker: str) -> str:
    """BRK.B -> BRK-B (formato de Yahoo para clases de acciones)."""
    return str(ticker).strip().upper().replace(".", "-").replace(" ", "-")


def parsear_ssga(contenido: bytes) -> pd.DataFrame:
    """Tabla de tenencias de SPY: ticker (Yahoo), nombre, sector, peso (fracción) y fecha del archivo."""
    crudo = pd.read_excel(io.BytesIO(contenido), header=None)
    fila = next((i for i, r in crudo.iterrows() if "Ticker" in {str(v).strip() for v in r.values}), None)
    if fila is None:
        raise ValueError("SSGA: no se encontró la fila de encabezados")
    fecha = None
    for _, r in crudo.iloc[:fila].iterrows():
        texto = " ".join(str(v) for v in r.values if pd.notna(v))
        if "As of" in texto:
            fecha = pd.to_datetime(texto.split("As of")[-1].strip(), errors="coerce")
    tabla = crudo.iloc[fila + 1:].copy()
    tabla.columns = [str(c).strip() for c in crudo.iloc[fila]]
    tabla["peso"] = pd.to_numeric(tabla.get("Weight"), errors="coerce") / 100
    tabla = tabla[tabla["Ticker"].notna() & tabla["peso"].notna() & (tabla["peso"] > 0)]
    tabla = tabla[~tabla["Ticker"].astype(str).str.contains("CASH|USD|^-$", regex=True)]
    salida = pd.DataFrame({
        "ticker": [a_yahoo(t) for t in tabla["Ticker"]],
        "nombre": tabla.get("Name", pd.Series("", index=tabla.index)).astype(str).to_numpy(),
        "sector": tabla.get("Sector", pd.Series("", index=tabla.index)).astype(str).to_numpy(),
        "peso": tabla["peso"].to_numpy(dtype=float),
    })
    salida["fecha"] = fecha if fecha is not None and pd.notna(fecha) else pd.Timestamp(date.today())
    salida["fuente"] = "SSGA (tenencias de SPY)"
    return salida.drop_duplicates("ticker").reset_index(drop=True)


def parsear_wikipedia(html: str) -> pd.DataFrame:
    tablas = pd.read_html(io.StringIO(html))
    t = next(t for t in tablas if "Symbol" in t.columns)
    salida = pd.DataFrame({
        "ticker": [a_yahoo(s) for s in t["Symbol"]],
        "nombre": t.get("Security", pd.Series("", index=t.index)).astype(str).to_numpy(),
        "sector": t.get("GICS Sector", pd.Series("", index=t.index)).astype(str).to_numpy(),
        "peso": np.nan,
    })
    salida["fecha"] = pd.Timestamp(date.today())
    salida["fuente"] = "Wikipedia (sin pesos)"
    return salida.drop_duplicates("ticker").reset_index(drop=True)


def componentes(timeout: int = 60) -> pd.DataFrame:
    """Componentes actuales; prueba SSGA y si falla usa Wikipedia. Vacío si fallan las dos."""
    try:
        r = requests.get(URL_SSGA, headers=CABECERAS, timeout=timeout)
        r.raise_for_status()
        t = parsear_ssga(r.content)
        if len(t) >= 400:
            return t
        log.warning("SSGA: solo %d componentes, se prueba Wikipedia", len(t))
    except Exception as e:  # noqa: BLE001
        log.warning("SSGA falló: %s", e)
    try:
        r = requests.get(URL_WIKI, headers=CABECERAS, timeout=timeout)
        r.raise_for_status()
        return parsear_wikipedia(r.text)
    except Exception as e:  # noqa: BLE001
        log.warning("Wikipedia falló: %s", e)
    return pd.DataFrame(columns=["ticker", "nombre", "sector", "peso", "fecha", "fuente"])
