"""Orquestación de la descarga de todas las fuentes."""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Proyecto
from .almacen import Almacen
from .calidad import reporte_precios
from .proveedores import calendario as calendario_oficial
from .proveedores import cboe, fred, french, sp500, yahoo

log = logging.getLogger(__name__)


def actualizar_todo(proyecto: Proyecto) -> dict[str, list[str]]:
    """Descarga todas las fuentes y devuelve {fuente: fallidos}."""
    alm = Almacen(proyecto.dir_datos)
    fallos: dict[str, list[str]] = {}

    tickers = proyecto.tickers_yahoo()
    log.info("Yahoo: %d tickers", len(tickers))
    tablas, fallidos = yahoo.descargar(tickers)
    for campo, tabla in tablas.items():
        alm.guardar(f"precios/{campo}", tabla)
    alm.registrar_descarga("yahoo", origen="descarga", url="Yahoo Finance (yfinance)", tickers=len(tickers), fallidos=fallidos)
    fallos["yahoo"] = fallidos

    if proyecto.sp500_activo:
        fallos["sp500"] = actualizar_sp500(proyecto, alm)

    series = list(proyecto.series_fred)
    log.info("FRED: %d series", len(series))
    tabla, fallidas = fred.descargar(series)
    if not tabla.empty:
        alm.guardar("fred", tabla)
    alm.registrar_descarga("fred", origen="descarga", url=fred.URL.split("?")[0], series=len(series), fallidos=fallidas)
    fallos["fred"] = fallidas

    log.info("CBOE: %d series", len(proyecto.series_cboe))
    tabla, fallidos_cboe = cboe.descargar(proyecto.series_cboe)
    if not tabla.empty:
        alm.guardar("cboe", tabla)
    alm.registrar_descarga("cboe", origen="descarga", url="https://cdn.cboe.com", series=len(proyecto.series_cboe), fallidos=fallidos_cboe)
    fallos["cboe"] = fallidos_cboe

    fallos["ken_french"] = []
    for nombre in proyecto.ken_french:
        try:
            alm.guardar(f"french/{nombre}", french.descargar(nombre))
        except Exception as e:  # noqa: BLE001
            log.warning("Ken French %s: %s", nombre, e)
            fallos["ken_french"].append(nombre)
    alm.registrar_descarga("ken_french", origen="descarga", url="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/",
                           series=len(proyecto.ken_french), fallidos=fallos["ken_french"])

    eventos, fallidas_cal = calendario_oficial.descargar()
    if not eventos.empty:
        eventos.to_csv(proyecto.dir_datos / "calendario_eventos.csv", index=False)
        alm.registrar_descarga("calendario", origen="descarga", url="federalreserve.gov, bls.gov",
                               eventos=len(eventos), fallidos=fallidas_cal)
    fallos["calendario"] = fallidas_cal

    cierre = alm.precios("cierre_aj")
    if "SPY" in cierre.columns:
        calendario = pd.DatetimeIndex(cierre["SPY"].dropna().index)
        asinc = {t for t in cierre.columns if proyecto.es_asincronico(t)}
        reporte_precios(cierre, calendario, asinc).to_csv(proyecto.dir_datos / "calidad_precios.csv")

    return fallos


def actualizar_sp500(proyecto: Proyecto, alm: Almacen) -> list[str]:
    """Componentes actuales del S&P 500 y sus precios. Devuelve los tickers que fallaron."""
    comp = sp500.componentes()
    if comp.empty:
        log.warning("S&P 500: no se pudieron obtener los componentes")
        return ["componentes"]
    comp.to_csv(proyecto.dir_datos / "sp500_componentes.csv", index=False)
    tickers = list(comp["ticker"])
    log.info("S&P 500: %d componentes (%s)", len(tickers), comp["fuente"].iloc[0])
    tablas, fallidos = yahoo.descargar(tickers)
    for campo in ("cierre_aj", "volumen"):
        if campo in tablas:
            alm.guardar(f"sp500/{campo}", tablas[campo])
    alm.registrar_descarga("sp500", origen="descarga", url=comp["fuente"].iloc[0], tickers=len(tickers),
                           fallidos=fallidos)
    return fallidos
