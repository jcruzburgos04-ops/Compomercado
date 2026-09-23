"""Precios diarios desde Yahoo Finance (vía yfinance)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

CAMPOS = {
    "Adj Close": "cierre_aj",
    "Close": "cierre",
    "Open": "apertura",
    "High": "maximo",
    "Low": "minimo",
    "Volume": "volumen",
}

NY = ZoneInfo("America/New_York")


def _descargar_lote(tickers: list[str], reintentos: int, pausa: float) -> pd.DataFrame:
    import yfinance as yf

    ultimo_error: Exception | None = None
    for intento in range(reintentos):
        try:
            df = yf.download(
                tickers,
                period="max",
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
                group_by="column",
                multi_level_index=True,
                timeout=30,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001 - yfinance lanza excepciones heterogéneas
            ultimo_error = e
            log.warning("Yahoo: intento %d falló para %d tickers: %s", intento + 1, len(tickers), e)
        time.sleep(pausa * (2**intento))
    if ultimo_error:
        log.error("Yahoo: lote sin datos tras %d intentos: %s", reintentos, ultimo_error)
    return pd.DataFrame()


def _separar_campos(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    salida: dict[str, pd.DataFrame] = {}
    if df.empty:
        return salida
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Se esperaban columnas MultiIndex (campo, ticker) de yfinance")
    nivel_campos = set(df.columns.get_level_values(0))
    for campo_yf, campo in CAMPOS.items():
        if campo_yf in nivel_campos:
            sub = df[campo_yf].copy()
            sub.columns = [str(c) for c in sub.columns]
            salida[campo] = sub
    return salida


def quitar_barra_incompleta(df: pd.DataFrame, ahora: datetime | None = None) -> pd.DataFrame:
    """Elimina la fila de hoy si la rueda de EE. UU. todavía no cerró (evita datos intradía)."""
    if df.empty:
        return df
    ahora = ahora or datetime.now(NY)
    hoy = pd.Timestamp(ahora.date())
    cierre = ahora.replace(hour=16, minute=30, second=0, microsecond=0)
    if ahora < cierre and df.index.max() >= hoy:
        return df.loc[df.index < hoy]
    return df


def descargar(
    tickers: list[str], lote: int = 40, reintentos: int = 3, pausa: float = 2.0
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Descarga el historial completo. Devuelve {campo: tabla ancha} y la lista de tickers fallidos."""
    partes: dict[str, list[pd.DataFrame]] = {c: [] for c in CAMPOS.values()}
    for i in range(0, len(tickers), lote):
        grupo = tickers[i : i + lote]
        campos = _separar_campos(_descargar_lote(grupo, reintentos, pausa))
        for campo, tabla in campos.items():
            partes[campo].append(tabla)

    tablas: dict[str, pd.DataFrame] = {}
    for campo, lista in partes.items():
        if not lista:
            continue
        tabla = pd.concat(lista, axis=1)
        tabla = tabla.loc[:, ~tabla.columns.duplicated()]
        tabla.index = pd.DatetimeIndex(tabla.index).tz_localize(None).normalize()
        tabla = tabla[~tabla.index.duplicated(keep="last")].sort_index()
        tablas[campo] = quitar_barra_incompleta(tabla.dropna(how="all", axis=1))

    base = tablas.get("cierre", pd.DataFrame())
    fallidos = [t for t in tickers if t not in base.columns or base[t].dropna().empty]

    # Reintento individual de los que fallaron en lote.
    if fallidos:
        log.info("Yahoo: reintentando %d tickers de a uno", len(fallidos))
        recuperados = []
        for t in fallidos:
            campos = _separar_campos(_descargar_lote([t], reintentos, pausa))
            if "cierre" in campos and not campos["cierre"].dropna(how="all").empty:
                recuperados.append(t)
                for campo, tabla in campos.items():
                    tabla.index = pd.DatetimeIndex(tabla.index).tz_localize(None).normalize()
                    tabla = quitar_barra_incompleta(tabla[~tabla.index.duplicated(keep="last")])
                    tablas[campo] = tablas.get(campo, pd.DataFrame()).join(tabla[[t]], how="outer")
        fallidos = [t for t in fallidos if t not in recuperados]

    return tablas, fallidos
