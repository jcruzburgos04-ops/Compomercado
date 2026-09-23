"""Fechas oficiales de eventos macro: reuniones de la Fed (FOMC) y publicaciones del BLS (CPI y empleo).

Se leen de las páginas oficiales. Si una página cambia de formato o no responde, ese evento
simplemente no aparece (se avisa en el log); nunca se inventan fechas.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL_FOMC = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
URL_BLS = {
    "cpi": "https://www.bls.gov/schedule/news_release/cpi.htm",
    "empleo": "https://www.bls.gov/schedule/news_release/empsit.htm",
}
NOMBRES_BLS = {"cpi": "Inflación (CPI)", "empleo": "Empleo (nóminas no agrícolas)"}
CABECERAS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
             "Accept-Language": "en-US,en;q=0.9"}

MESES = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                     "september", "october", "november", "december"], start=1)}
MESES_CORTOS = {m[:3]: i for m, i in MESES.items()}


def _texto(html: str) -> str:
    """HTML a texto plano con separadores, suficiente para buscar patrones."""
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " | ", html))


def parsear_fomc(html: str) -> pd.DataFrame:
    """Día de decisión (último día) de cada reunión de la Fed publicada en el calendario oficial."""
    texto = _texto(html)
    filas = []
    bloques = re.split(r"(\d{4}) FOMC Meetings", texto)
    for i in range(1, len(bloques) - 1, 2):
        anio, cuerpo = int(bloques[i]), bloques[i + 1]
        patron = (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"(?:/(January|February|March|April|May|June|July|August|September|October|November|December))?"
                  r"\s*\|[\s|]*(\d{1,2})(?:-(\d{1,2}))?")
        for m in re.finditer(patron, cuerpo):
            mes1, mes2, d1, d2 = m.groups()
            mes = MESES[(mes2 or mes1).lower()]
            dia = int(d2 or d1)
            try:
                filas.append({"fecha": pd.Timestamp(anio, mes, dia), "evento": "Decisión de la Fed (FOMC)",
                              "tipo": "fed"})
            except ValueError:
                continue
    df = pd.DataFrame(filas, columns=["fecha", "evento", "tipo"])
    return df.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)


def parsear_bls(html: str, clave: str) -> pd.DataFrame:
    """Fechas de publicación de la tabla de calendario del BLS ("Oct. 15, 2026").

    Solo se leen las tablas: afuera hay otras fechas (por ejemplo, la de última modificación)."""
    tablas = re.findall(r"(?is)<table.*?</table>", html)
    texto = _texto(" ".join(tablas) if tablas else html)
    filas = []
    for mes, dia, anio in re.findall(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2}),\s+(\d{4})",
                                     texto):
        try:
            filas.append({"fecha": pd.Timestamp(int(anio), MESES_CORTOS[mes[:3].lower()], int(dia)),
                          "evento": NOMBRES_BLS[clave], "tipo": clave})
        except ValueError:
            continue
    df = pd.DataFrame(filas, columns=["fecha", "evento", "tipo"])
    return df.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)


def descargar(timeout: int = 30) -> tuple[pd.DataFrame, list[str]]:
    """Todas las fechas oficiales disponibles y la lista de fuentes que fallaron."""
    partes, fallidas = [], []
    try:
        r = requests.get(URL_FOMC, headers=CABECERAS, timeout=timeout)
        r.raise_for_status()
        df = parsear_fomc(r.text)
        if df.empty:
            raise ValueError("sin reuniones reconocibles")
        partes.append(df)
        log.info("Calendario: %d reuniones de la Fed (%s a %s)", len(df), df["fecha"].min().date(), df["fecha"].max().date())
    except Exception as e:  # noqa: BLE001
        log.warning("Calendario de la Fed falló: %s", e)
        fallidas.append("fomc")
    for clave, url in URL_BLS.items():
        try:
            r = requests.get(url, headers=CABECERAS, timeout=timeout)
            r.raise_for_status()
            df = parsear_bls(r.text, clave)
            if df.empty:
                raise ValueError("sin fechas reconocibles")
            partes.append(df)
            log.info("Calendario: %d fechas de %s", len(df), clave)
        except Exception as e:  # noqa: BLE001
            log.warning("Calendario BLS %s falló: %s", clave, e)
            fallidas.append(clave)
    datos = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=["fecha", "evento", "tipo"])
    return datos, fallidas
