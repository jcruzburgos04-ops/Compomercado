"""Kenneth French Data Library: industrias y factores diarios desde 1926."""

from __future__ import annotations

import io
import logging
import re
import zipfile

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{nombre}_CSV.zip"
UA = {"User-Agent": "Mozilla/5.0 (compomercado)"}


def parsear_bloque(texto: str, titulo: str | None = None) -> pd.DataFrame:
    """Parsea el primer bloque de datos diarios (fecha AAAAMMDD) del CSV de Ken French.

    Si se indica `titulo` (regex), el bloque se busca después de la línea que lo contiene.
    Los valores vienen en porcentaje y -99.99 / -999 indican dato faltante.
    """
    lineas = texto.splitlines()
    inicio = 0
    if titulo:
        patron = re.compile(titulo, re.IGNORECASE)
        for i, linea in enumerate(lineas):
            if patron.search(linea):
                inicio = i + 1
                break
    encabezado = None
    for i in range(inicio, len(lineas)):
        if lineas[i].strip().startswith(","):
            encabezado = i
            break
    if encabezado is None:
        raise ValueError("No se encontró el encabezado del bloque")
    columnas = [c.strip() for c in lineas[encabezado].split(",")][1:]
    fechas, filas = [], []
    for linea in lineas[encabezado + 1 :]:
        partes = [p.strip() for p in linea.split(",")]
        if not partes or not re.fullmatch(r"\d{8}", partes[0]):
            break
        fechas.append(partes[0])
        filas.append(partes[1 : len(columnas) + 1])
    df = pd.DataFrame(filas, index=pd.to_datetime(fechas, format="%Y%m%d"), columns=columnas)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= -99.99)
    df.index.name = "fecha"
    return df / 100.0


def descargar(nombre: str) -> pd.DataFrame:
    r = requests.get(URL.format(nombre=nombre), headers=UA, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"{nombre}: el zip no contiene CSV")
        texto = z.read(csvs[0]).decode("latin-1")
    titulo = r"Value Weighted Returns" if "Industry" in nombre else None
    return parsear_bloque(texto, titulo)
