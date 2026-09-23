"""Screener sobre el mapa de comportamiento (config/screener.yaml)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def aplicar(tabla: pd.DataFrame, screens: dict, contexto: dict) -> list[dict]:
    """Corre cada screen y devuelve [{clave, titulo, descripcion, estado, resultado}].

    estado: "ok" | "no_aplica" (no se cumple `solo_si`) | "error" (consulta inválida).
    """
    salida = []
    for clave, cfg in (screens or {}).items():
        cfg = cfg or {}
        item = {"clave": clave, "titulo": cfg.get("titulo", clave), "descripcion": cfg.get("descripcion", ""),
                "consulta": cfg.get("consulta", ""), "estado": "ok", "resultado": pd.DataFrame()}
        try:
            condicion = cfg.get("solo_si")
            if condicion and not bool(pd.eval(condicion, local_dict=contexto)):
                item["estado"] = "no_aplica"
                salida.append(item)
                continue
            t = tabla
            if cfg.get("grupos"):
                t = t[t["grupo"].isin(cfg["grupos"])]
            if item["consulta"]:
                t = t.query(item["consulta"], local_dict={})
            orden = cfg.get("orden")
            if orden in t.columns:
                t = t.sort_values(orden, ascending=bool(cfg.get("ascendente", False)))
            item["resultado"] = t.head(int(cfg.get("limite", 20)))
        except Exception as e:  # noqa: BLE001 - un screen mal escrito no frena el resto
            log.warning("Screen %s inválido: %s", clave, e)
            item["estado"] = "error"
            item["error"] = str(e)
        salida.append(item)
    return salida


def correlacion_interna(precios: pd.DataFrame, tickers: list[str], ventanas=(63, 252)) -> dict:
    """Correlación promedio entre pares de los componentes de una canasta en cada ventana."""
    cols = [t for t in tickers if t in precios.columns]
    if len(cols) < 3:
        return {}
    r = precios[cols].pct_change(fill_method=None)
    salida = {"n": len(cols)}
    for h in ventanas:
        bloque = r.tail(h).dropna(axis=1, how="any")
        if bloque.shape[1] < 3:
            continue
        c = np.corrcoef(bloque.to_numpy(), rowvar=False)
        k = c.shape[0]
        salida[f"corr_{h}"] = float((c.sum() - k) / (k * (k - 1)))
    return salida
