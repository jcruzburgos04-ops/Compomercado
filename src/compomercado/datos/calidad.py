"""Controles de calidad sobre las tablas de precios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reporte_precios(cierre_aj: pd.DataFrame, calendario: pd.DatetimeIndex, asincronicos: set[str]) -> pd.DataFrame:
    """Una fila por ticker con inicio, fin, huecos, saltos anómalos y precios congelados.

    `calendario` son las ruedas de EE. UU. (fechas de SPY). Para activos sincrónicos, un hueco
    es una rueda de EE. UU. sin dato dentro de la vida del activo.
    """
    filas = []
    ultima_rueda = calendario.max() if len(calendario) else None
    for t in cierre_aj.columns:
        s = cierre_aj[t].dropna()
        if s.empty:
            filas.append({"ticker": t, "estado": "sin datos"})
            continue
        r = s.pct_change().dropna()
        vida = calendario[(calendario >= s.index.min()) & (calendario <= s.index.max())]
        huecos = np.nan if t in asincronicos else int(len(vida.difference(s.index)))
        congelado = int((s.diff() == 0).astype(int).groupby((s.diff() != 0).cumsum()).sum().max() or 0)
        salto = float(r.abs().max()) if len(r) else np.nan
        atraso = int(len(calendario[(calendario > s.index.max())])) if ultima_rueda is not None else 0
        avisos = []
        if atraso > 3:
            avisos.append(f"sin datos en las últimas {atraso} ruedas")
        if salto > 0.4:
            avisos.append(f"salto diario de {salto:.0%}")
        if congelado >= 10:
            avisos.append(f"precio congelado {congelado} días")
        if isinstance(huecos, int) and len(vida) and huecos / len(vida) > 0.02:
            avisos.append(f"{huecos} huecos")
        filas.append(
            {
                "ticker": t,
                "inicio": s.index.min().date(),
                "fin": s.index.max().date(),
                "observaciones": len(s),
                "huecos": huecos,
                "mayor_salto": salto,
                "dias_congelado": congelado,
                "estado": "; ".join(avisos) if avisos else "ok",
            }
        )
    return pd.DataFrame(filas).set_index("ticker")
