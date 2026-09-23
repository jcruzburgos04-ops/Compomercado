"""Carga de la configuración (universos, canastas, parámetros) y rutas del proyecto."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Instrumento:
    ticker: str
    nombre: str
    grupo: str
    zona: str | None = None


@dataclass(frozen=True)
class Canasta:
    nombre: str
    descripcion: str
    ponderacion: str
    rebalanceo: str
    componentes: tuple[tuple[str, float | None], ...]

    @property
    def tickers(self) -> list[str]:
        return [t for t, _ in self.componentes]


@dataclass
class Proyecto:
    """Rutas y configuración de una instancia del proyecto."""

    raiz: Path = field(default_factory=lambda: Path(os.environ.get("COMPOMERCADO_RAIZ", Path.cwd())))

    @property
    def dir_config(self) -> Path:
        return self.raiz / "config"

    @property
    def dir_datos(self) -> Path:
        return self.raiz / "datos"

    @property
    def dir_registro(self) -> Path:
        return self.raiz / "registro"

    @property
    def dir_sitio(self) -> Path:
        return self.raiz / "sitio"

    def _yaml(self, nombre: str) -> dict:
        ruta = self.dir_config / nombre
        if not ruta.exists():
            return {}
        with ruta.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @cached_property
    def universos(self) -> dict:
        return self._yaml("universos.yaml")

    @cached_property
    def analisis(self) -> dict:
        return self._yaml("analisis.yaml")

    @cached_property
    def instrumentos(self) -> list[Instrumento]:
        salida = []
        for grupo, valor in self.universos.items():
            if not isinstance(valor, list) or not valor or not isinstance(valor[0], dict):
                continue
            for item in valor:
                if "ticker" not in item:
                    continue
                salida.append(
                    Instrumento(
                        ticker=str(item["ticker"]),
                        nombre=str(item.get("nombre", item["ticker"])),
                        grupo=grupo,
                        zona=item.get("zona"),
                    )
                )
        return salida

    def grupo(self, nombre: str) -> list[Instrumento]:
        return [i for i in self.instrumentos if i.grupo == nombre]

    @cached_property
    def nombres(self) -> dict[str, str]:
        return {i.ticker: i.nombre for i in self.instrumentos}

    @cached_property
    def grupo_de(self) -> dict[str, str]:
        salida: dict[str, str] = {}
        for i in self.instrumentos:
            salida.setdefault(i.ticker, i.grupo)
        return salida

    @cached_property
    def canastas(self) -> list[Canasta]:
        salida = []
        for nombre, c in self._yaml("canastas.yaml").items():
            componentes = []
            for comp in c.get("componentes") or []:
                if isinstance(comp, dict):
                    componentes.append((str(comp["ticker"]), comp.get("peso")))
                else:
                    componentes.append((str(comp), None))
            salida.append(
                Canasta(
                    nombre=nombre,
                    descripcion=c.get("descripcion", nombre),
                    ponderacion=c.get("ponderacion", "igual"),
                    rebalanceo=c.get("rebalanceo", "mensual"),
                    componentes=tuple(componentes),
                )
            )
        return salida

    @property
    def series_fred(self) -> dict[str, dict]:
        return self.universos.get("fred", {}) or {}

    @property
    def series_cboe(self) -> list[str]:
        return list(self.universos.get("cboe", []) or [])

    @property
    def argentina(self) -> dict:
        return self.universos.get("argentina", {}) or {}

    @property
    def ken_french(self) -> list[str]:
        return list(self.universos.get("ken_french", []) or [])

    def tickers_yahoo(self) -> list[str]:
        """Todos los símbolos a descargar de Yahoo: universo + canastas + Argentina."""
        tickers = [i.ticker for i in self.instrumentos]
        for c in self.canastas:
            tickers.extend(c.tickers)
        arg = self.argentina
        if arg.get("merval"):
            tickers.append(arg["merval"])
        for par in arg.get("ccl_pares", []) or []:
            tickers.extend([par["adr"], par["local"]])
        vistos: dict[str, None] = {}
        for t in tickers:
            vistos.setdefault(t, None)
        return list(vistos)

    def es_asincronico(self, ticker: str) -> bool:
        """True si el activo no cotiza en el horario de EE. UU. (se analiza con retornos semanales)."""
        grupos = set(self.analisis.get("grupos_asincronicos", []))
        if self.grupo_de.get(ticker) in grupos:
            return True
        return ticker.endswith(("=F", "=X", "-USD"))
