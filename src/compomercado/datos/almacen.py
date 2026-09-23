"""Almacén local en Parquet: tablas anchas (fecha x serie) y metadatos de cada descarga."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

CAMPOS_PRECIO = ("cierre_aj", "cierre", "apertura", "maximo", "minimo", "volumen")


class Almacen:
    def __init__(self, directorio: Path):
        self.dir = Path(directorio)

    def _ruta(self, nombre: str) -> Path:
        return self.dir / f"{nombre}.parquet"

    def guardar(self, nombre: str, df: pd.DataFrame) -> None:
        ruta = self._ruta(nombre)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index, name="fecha")
        df.columns = [str(c) for c in df.columns]
        df.sort_index().to_parquet(ruta)

    def existe(self, nombre: str) -> bool:
        return self._ruta(nombre).exists()

    def leer(self, nombre: str) -> pd.DataFrame:
        ruta = self._ruta(nombre)
        if not ruta.exists():
            return pd.DataFrame(index=pd.DatetimeIndex([], name="fecha"))
        return pd.read_parquet(ruta)

    # --- precios ---------------------------------------------------------
    def precios(self, campo: str = "cierre_aj", tickers: list[str] | None = None) -> pd.DataFrame:
        df = self.leer(f"precios/{campo}")
        if tickers is not None:
            df = df.reindex(columns=[t for t in tickers if t in df.columns])
        return df

    def ohlc_ajustado(self, ticker: str) -> pd.DataFrame:
        """OHLC ajustado por dividendos y splits (factor = cierre ajustado / cierre)."""
        cols = {}
        for campo in CAMPOS_PRECIO:
            df = self.leer(f"precios/{campo}")
            if ticker in df.columns:
                cols[campo] = df[ticker]
        out = pd.DataFrame(cols).dropna(subset=["cierre"])
        factor = (out["cierre_aj"] / out["cierre"]).fillna(1.0)
        for campo in ("apertura", "maximo", "minimo"):
            if campo in out:
                out[campo] = out[campo] * factor
        out["cierre"] = out["cierre_aj"]
        return out.drop(columns=["cierre_aj"])

    # --- metadatos -------------------------------------------------------
    def _ruta_meta(self) -> Path:
        return self.dir / "metadatos.json"

    def metadatos(self) -> dict:
        ruta = self._ruta_meta()
        if not ruta.exists():
            return {}
        return json.loads(ruta.read_text(encoding="utf-8"))

    def registrar_descarga(self, fuente: str, **info) -> None:
        meta = self.metadatos()
        meta[fuente] = {"actualizado_utc": datetime.now(UTC).isoformat(timespec="seconds"), **info}
        self._ruta_meta().parent.mkdir(parents=True, exist_ok=True)
        self._ruta_meta().write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
