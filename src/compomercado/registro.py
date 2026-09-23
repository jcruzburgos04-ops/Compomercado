"""Registro forward: una fila por día, solo se agregan filas nuevas (nunca se reescriben las pasadas)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import __version__


def agregar(directorio: Path, archivo: str, fecha, valores: dict) -> bool:
    """Agrega la fila de `fecha` si todavía no existe. Devuelve True si agregó."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / archivo
    fecha = pd.Timestamp(fecha).strftime("%Y-%m-%d")
    fila = {"fecha": fecha, "registrado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "version": __version__, **valores}
    nueva = pd.DataFrame([fila])
    if ruta.exists():
        existente = pd.read_csv(ruta, dtype={"fecha": str})
        if fecha in set(existente["fecha"]):
            return False
        # Columnas nuevas quedan vacías en las filas viejas; los valores pasados no se tocan.
        tabla = pd.concat([existente, nueva], ignore_index=True)
    else:
        tabla = nueva
    tabla.to_csv(ruta, index=False)
    return True


def leer(directorio: Path, archivo: str) -> pd.DataFrame:
    ruta = directorio / archivo
    if not ruta.exists():
        return pd.DataFrame()
    df = pd.read_csv(ruta, dtype={"fecha": str})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha")
