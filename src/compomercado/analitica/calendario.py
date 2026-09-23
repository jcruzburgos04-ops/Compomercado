"""Pilar 10, calendario: eventos que cambian el comportamiento del mercado por unos días.

Es un modulador, no una señal: no entra al puntaje. Sirve para saber qué viene (vencimientos de
opciones, del VIX, fin de mes o de trimestre, Fed, inflación, empleo) y cómo se portó
históricamente el mes en curso.

Las fechas de vencimientos y fines de mes salen de las reglas de la bolsa; las de la Fed y el
BLS, de sus calendarios oficiales (ver `datos/proveedores/calendario.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay


class FeriadosNYSE(AbstractHolidayCalendar):
    rules = [
        Holiday("Año nuevo", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independencia", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Navidad", month=12, day=25, observance=nearest_workday),
    ]


RUEDA = CustomBusinessDay(calendar=FeriadosNYSE())


def es_rueda(fecha: pd.Timestamp) -> bool:
    return RUEDA.is_on_offset(pd.Timestamp(fecha))


def rueda_anterior_o_igual(fecha: pd.Timestamp) -> pd.Timestamp:
    fecha = pd.Timestamp(fecha)
    return fecha if es_rueda(fecha) else fecha - RUEDA


def tercer_viernes(anio: int, mes: int) -> pd.Timestamp:
    primero = pd.Timestamp(anio, mes, 1)
    viernes = primero + pd.Timedelta(days=(4 - primero.weekday()) % 7)
    return viernes + pd.Timedelta(weeks=2)


def opex(anio: int, mes: int) -> pd.Timestamp:
    """Vencimiento mensual de opciones: tercer viernes (o la rueda anterior si es feriado)."""
    return rueda_anterior_o_igual(tercer_viernes(anio, mes))


def vencimiento_vix(anio: int, mes: int) -> pd.Timestamp:
    """Vencimiento del VIX del mes: miércoles 30 días antes del tercer viernes del mes siguiente."""
    sig = pd.Timestamp(anio, mes, 1) + pd.DateOffset(months=1)
    viernes = rueda_anterior_o_igual(tercer_viernes(sig.year, sig.month))
    return rueda_anterior_o_igual(viernes - pd.Timedelta(days=30))


def fin_de_mes(anio: int, mes: int) -> pd.Timestamp:
    return rueda_anterior_o_igual(pd.Timestamp(anio, mes, 1) + pd.offsets.MonthEnd(0))


def eventos_reglas(desde, hasta) -> pd.DataFrame:
    """Vencimientos y fines de mes entre dos fechas."""
    desde, hasta = pd.Timestamp(desde), pd.Timestamp(hasta)
    filas = []
    for p in pd.period_range(desde - pd.DateOffset(months=1), hasta + pd.DateOffset(months=1), freq="M"):
        a, m = p.year, p.month
        trimestral = m in (3, 6, 9, 12)
        filas.append({"fecha": opex(a, m), "evento": "Vencimiento trimestral de opciones (triple witching)"
                      if trimestral else "Vencimiento mensual de opciones", "tipo": "vencimiento"})
        filas.append({"fecha": vencimiento_vix(a, m), "evento": "Vencimiento del VIX", "tipo": "vencimiento"})
        filas.append({"fecha": fin_de_mes(a, m), "evento": "Fin de trimestre (rebalanceo)" if trimestral
                      else "Fin de mes", "tipo": "fin_de_mes"})
    df = pd.DataFrame(filas)
    return df[(df["fecha"] >= desde) & (df["fecha"] <= hasta)].sort_values("fecha").reset_index(drop=True)


def proximos(fecha, oficiales: pd.DataFrame | None = None, dias: int = 30) -> pd.DataFrame:
    """Eventos de los próximos `dias` corridos, con las ruedas que faltan para cada uno."""
    fecha = pd.Timestamp(fecha)
    hasta = fecha + pd.Timedelta(days=dias)
    partes = [eventos_reglas(fecha + pd.Timedelta(days=1), hasta)]
    if oficiales is not None and not oficiales.empty:
        o = oficiales.copy()
        o["fecha"] = pd.to_datetime(o["fecha"])
        partes.append(o[(o["fecha"] > fecha) & (o["fecha"] <= hasta)])
    df = pd.concat(partes, ignore_index=True).sort_values(["fecha", "evento"]).reset_index(drop=True)
    df["ruedas"] = [len(pd.date_range(fecha, f, freq=RUEDA)) - 1 for f in df["fecha"]]
    return df


def estacionalidad(p: pd.Series, fecha, h: int = 21) -> dict:
    """Cómo se portó el índice en el mismo mes y en las próximas `h` ruedas desde la misma fecha del
    año, con los años anteriores al actual (point-in-time)."""
    p = p.dropna()
    fecha = pd.Timestamp(fecha)
    if p.empty:
        return {}
    mensual = p.resample("ME").last().pct_change().dropna()
    previos = mensual[(mensual.index.month == fecha.month) & (mensual.index.year < fecha.year)]
    adelante = []
    for anio in sorted(set(p.index.year)):
        if anio >= fecha.year:
            continue
        try:
            inicio = p.index[p.index.searchsorted(fecha.replace(year=anio))]
        except (IndexError, ValueError):
            continue
        i = p.index.get_loc(inicio)
        if i + h < len(p):
            adelante.append(p.iloc[i + h] / p.iloc[i] - 1)
    adelante = np.array(adelante)
    return {
        "mes": fecha.month,
        "anios": int(len(previos)),
        "mes_medio": float(previos.mean()) if len(previos) else np.nan,
        "mes_positivo": float((previos > 0).mean()) if len(previos) else np.nan,
        "adelante_medio": float(adelante.mean()) if len(adelante) else np.nan,
        "adelante_positivo": float((adelante > 0).mean()) if len(adelante) else np.nan,
        "adelante_n": int(len(adelante)),
        "todos_medio": float(mensual[mensual.index.year < fecha.year].mean()),
    }
