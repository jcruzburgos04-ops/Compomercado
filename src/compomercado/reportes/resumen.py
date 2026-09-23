"""Parte diario en texto (log de la corrida y resumen de GitHub Actions)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicadores.estado import PILARES
from ..pipeline import Resultados


def _pct(v: float) -> str:
    return "—" if v is None or pd.isna(v) else f"{v * 100:+.1f}%"


def texto(res: Resultados) -> str:
    L: list[str] = [f"COMPOMERCADO · datos al cierre del {res.fecha:%Y-%m-%d}"]
    o = res.origen or {}
    if o.get("real"):
        L.append(f"Origen: descarga real de {', '.join(o['fuentes'])} ({str(o.get('descargado_utc'))[:16]} UTC)")
    else:
        L.append("ATENCIÓN: DATOS DE PRUEBA, no reflejan el mercado")
    if res.ultimas_fechas:
        L.append("Último dato: " + " · ".join(f"{k} {v:%d/%m}" for k, v in res.ultimas_fechas.items()))
    L.append("")
    pil = res.pilares
    pond = res.ponderacion
    vig = pond.pesos_vigentes if pond is not None else None
    if "total" in pil and pil["total"].notna().any():
        t = pil["total"].dropna()
        previo = f"{t.iloc[-1 - 5]:.0f}" if len(t) > 5 else "—"
        igual = res.pilares_igual["total"].dropna() if "total" in res.pilares_igual else pd.Series(dtype=float)
        etiqueta = "ponderado" if vig is not None else "pesos iguales"
        extra = f" · con pesos iguales {igual.iloc[-1]:.0f}" if vig is not None and len(igual) else ""
        L.append(f"RIESGO TOTAL ({etiqueta}): {t.iloc[-1]:.0f}/100 · hace 5 ruedas {previo}{extra}")
        for p, nombre in PILARES.items():
            if p in pil and pil[p].notna().any():
                s = pil[p].dropna()
                peso = f"  peso {pond.peso_pilar_total.get(p, 0):.0%}" if pond is not None else ""
                L.append(f"  {nombre:<28}{s.iloc[-1]:5.0f}{peso}")
        ult = res.riesgo.ffill().iloc[-1].dropna().sort_values()
        if vig is not None:
            ult = ult[[i for i in ult.index if vig.indicadores["peso"].get(i, 0) > 0]]
        ids = {i.id: i for i in res.indicadores}
        fmt = lambda k: f"{ids[k].nombre} p{ult[k]:.0f}"  # noqa: E731
        L.append("  Empujan: " + "; ".join(fmt(k) for k in ult.index[::-1][:5]))
        L.append("  Calman:  " + "; ".join(fmt(k) for k in ult.index[:5]))
        L.append("")

    if vig is not None:
        L.append(f"PONDERACIONES (pesos estimados con datos hasta {vig.hasta:%Y-%m-%d})")
        ti = vig.indicadores
        for p, nombre in PILARES.items():
            if p not in vig.pilares.index:
                continue
            fp = vig.pilares.loc[p]
            L.append(f"  {nombre} — peso {pond.peso_pilar_total.get(p, 0):.0%}, AUC {fp['auc_media']:.3f}, {fp['estado']}")
            sub = ti[ti["pilar"] == p].sort_values(["peso", "auc_media"], ascending=False)
            for i, f in sub.iterrows():
                marca = f"{f['peso']:4.0%}" if f["peso"] > 0 else "  — "
                L.append(f"      {marca}  AUC {f['auc_media']:.3f}  {ids[i].nombre if i in ids else i}"
                         + ("" if f["estado"] == "incluido" else f"  [{f['estado']}]"))
        L.append("")
        ev = pond.evaluacion
        L.append("EVALUACIÓN FUERA DE MUESTRA (AUC; 0,5 = no anticipa)")
        for per in ev["periodo"].unique():
            L.append(f"  {per}:")
            for _, f in ev[ev["periodo"] == per].iterrows():
                lift = f["tope20_y3"] / f["base_y3"] if f["base_y3"] else float("nan")
                L.append(f"    {f['variante']:<36} 5d {f['auc_y1']:.3f} · 10d {f['auc_y2']:.3f} · 21d {f['auc_y3']:.3f}"
                         f" · top 20%: {f['tope20_y3']:.0%} vs {f['base_y3']:.0%} ({lift:.2f}x)")
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
