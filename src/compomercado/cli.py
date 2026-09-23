"""Línea de comandos: `compomercado datos | analizar | todo`."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from .config import Proyecto

app = typer.Typer(help="Compomercado: sensor de comportamiento de mercado y sectores.", no_args_is_help=True)


def _proyecto(raiz: Path | None) -> Proyecto:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return Proyecto(raiz.resolve()) if raiz else Proyecto()


@app.command()
def datos(raiz: Path = typer.Option(None, help="Raíz del proyecto (por defecto, el directorio actual).")):
    """Descarga todas las fuentes (Yahoo, FRED, CBOE, Ken French)."""
    from .datos.actualizar import actualizar_todo

    fallos = actualizar_todo(_proyecto(raiz))
    for fuente, lista in fallos.items():
        typer.echo(f"{fuente}: {len(lista)} fallidos {lista if lista else ''}")


@app.command()
def analizar(
    raiz: Path = typer.Option(None, help="Raíz del proyecto."),
    registrar: bool = typer.Option(True, help="Agregar la fila del día al registro forward."),
    opciones: bool = typer.Option(True, help="Tomar snapshot de cadenas de opciones y ETFs."),
):
    """Corre el análisis y construye el dashboard en ./sitio."""
    from .pipeline import analizar as correr
    from .reportes.dashboard import construir
    from .reportes.resumen import texto

    p = _proyecto(raiz)
    res = correr(p, registrar=registrar, snapshot_opciones=opciones)
    ruta = construir(res, p.dir_sitio)
    parte = texto(res)
    typer.echo(parte)
    typer.echo(f"Dashboard: {ruta}")
    resumen_gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if resumen_gh:
        with open(resumen_gh, "a", encoding="utf-8") as f:
            f.write(f"## Parte diario\n\n```\n{parte}\n```\n")


@app.command()
def todo(raiz: Path = typer.Option(None, help="Raíz del proyecto.")):
    """Descarga datos, analiza y construye el dashboard."""
    datos(raiz)
    analizar(raiz, registrar=True, opciones=True)


if __name__ == "__main__":
    app()
