"""Parte diario en texto (log de la corrida y resumen de GitHub Actions)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicadores.estado import PILARES
from ..pipeline import Resultados


def _pct(v: float) -> str:
    return "—" if v is None or pd.isna(v) else f"{v * 100:+.1f}%"


def texto(res: Resultados) -> str:
    L: list[str] = [f"COMPOMERCADO · datos al cierre del {res.fecha:%Y-%m-%d}", ""]
    pil = res.pilares
    if "total" in pil and pil["total"].notna().any():
        t = pil["total"].dropna()
        previo = f"{t.iloc[-6]:.0f}" if len(t) > 5 else "—"
        L.append(f"RIESGO TOTAL (preliminar): {t.iloc[-1]:.0f}/100 · hace 5 ruedas {previo}")
        for p, nombre in PILARES.items():
            if p in pil and pil[p].notna().any():
                s = pil[p].dropna()
                L.append(f"  {nombre:<28}{s.iloc[-1]:5.0f}")
        ult = res.riesgo.ffill().iloc[-1].dropna().sort_values()
        ids = {i.id: i for i in res.indicadores}
        fmt = lambda k: f"{ids[k].nombre} p{ult[k]:.0f}"  # noqa: E731
        L.append("  Empujan: " + "; ".join(fmt(k) for k in ult.index[::-1][:5]))
        L.append("  Calman:  " + "; ".join(fmt(k) for k in ult.index[:5]))
        L.append("")

    spy = res.mapas.get("SPY")
    if spy is not None:
        tb = spy.tabla[(spy.tabla["frecuencia"] == "diaria") & ~spy.tabla.index.str.startswith("^")]
        acciones = tb[tb["grupo"].isin(["sectores", "industrias", "factores", "global_etf", "referencias", "canasta"])]
        fuera = tb[tb["grupo"].isin(["renta_fija", "commodities", "divisas", "cripto"])]
        cols = ["puntaje_refugio", "captura_mediana", "acierto_defensivo", "beta_bajista", "rs_63_pct"]
        L.append(f"MAPA VS SPY ({len(spy.episodios)} tramos ≥10% desde {spy.episodios['pico'].min():%Y})")
        L.append("  Refugio   captura  acierto  beta↓  RS63   activo")

        def filas(d):
            for tk, f in d[cols].iterrows():
                L.append(f"  {f.puntaje_refugio:6.0f}   {f.captura_mediana:6.2f}   {f.acierto_defensivo:6.0%}  "
                         f"{f.beta_bajista:5.2f}  {f.rs_63_pct:4.0f}   {tk} {res.nombres.get(tk, '')}")

        L.append("  Acciones, mejores refugios:")
        filas(acciones.sort_values("puntaje_refugio", ascending=False).head(10))
        L.append("  Acciones, lo que más cae:")
        filas(acciones.sort_values("puntaje_refugio").head(6).iloc[::-1])
        L.append("  Fuera de acciones:")
        filas(fuera.sort_values("puntaje_refugio", ascending=False).head(6))
        L.append("")
        L.append("ÚLTIMOS TRAMOS DE CAÍDA DE SPY")
        for _, e in spy.episodios.sort_values("pico").tail(6).iterrows():
            L.append(f"  {e['pico']:%Y-%m-%d} → {e['valle']:%Y-%m-%d}  {e['profundidad']:+.1%}  {e.get('tipo', '')}")
        L.append("")

    h = res.historia_larga
    if h is not None:
        tb = h.tabla.sort_values("puntaje_refugio", ascending=False)
        L.append(f"HISTORIA DESDE 1926 ({len(h.episodios)} mercados bajistas ≥20%)")
        L.append("  Mejores refugios: " + ", ".join(f"{i} ({v:.0f})" for i, v in tb["puntaje_refugio"].head(6).items()))
        L.append("  Peores:           " + ", ".join(f"{i} ({v:.0f})" for i, v in tb["puntaje_refugio"].tail(6).items()))
        L.append("")

    a = res.argentina
    if a:
        partes = []
        if "ccl" in a and a["ccl"].notna().any():
            partes.append(f"CCL {a['ccl'].dropna().iloc[-1]:,.0f}")
        if "merval_usd" in a and a["merval_usd"].notna().any():
            m = a["merval_usd"].dropna()
            partes.append(f"Merval USD {m.iloc[-1]:,.0f} ({_pct(m.iloc[-1] / m.iloc[-22] - 1 if len(m) > 22 else np.nan)} 21d)")
        if "exposicion" in a and a["exposicion"]["r2_global"].notna().any():
            partes.append(f"R² global {a['exposicion']['r2_global'].dropna().iloc[-1]:.0%}")
        L.append("ARGENTINA: " + " · ".join(partes))
        L.append("")

    L.append(f"Registro forward: {len(res.registro_estado)} ruedas · opciones: {len(res.registro_opciones)} snapshots")
    return "\n".join(L)
