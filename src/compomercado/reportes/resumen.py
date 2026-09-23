"""Parte diario en texto (log de la corrida y resumen de GitHub Actions)."""

from __future__ import annotations

import pandas as pd

from ..indicadores.estado import PILARES
from ..pipeline import Resultados


def _pct0(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.0%}"


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
        ind_sp = {i.id: i.serie.dropna() for i in res.indicadores if i.id.startswith("sp500_")}
        if res.sp500 and ind_sp:
            partes = [f"{ind_sp[k].iloc[-1]:.0%} sobre la media de {h}" for k, h in
                      (("sp500_pct_200", 200), ("sp500_pct_50", 50)) if k in ind_sp and len(ind_sp[k])]
            if res.sp500.get("top10") is not None:
                partes.append(f"top 10 = {res.sp500['top10']:.0%} del índice")
            L.append(f"  S&P 500 ({res.sp500['n']} empresas actuales, contexto): " + " · ".join(partes))
        L.append("")

    cal = res.calendario
    if cal and not cal["proximos"].empty:
        L.append("CALENDARIO (próximas 5 semanas; no suma al puntaje)")
        for _, f in cal["proximos"].iterrows():
            L.append(f"  {f['fecha']:%a %d/%m}  en {f['ruedas']:>2} ruedas  {f['evento']}")
        e = cal.get("estacionalidad") or {}
        if e.get("anios"):
            L.append(f"  Estacionalidad del mes: {e['mes_medio']:+.1%} promedio, positivo {e['mes_positivo']:.0%} de "
                     f"{e['anios']} años · próximas 21 ruedas: {e['adelante_medio']:+.1%}")
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
        L.append(f"MAPA VS SPY ({len(spy.episodios)} tramos ≥{res.umbral_tramos:.0%} desde {spy.episodios['pico'].min():%Y})")
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

    hu = res.huellas
    if hu is not None and not hu.tramos.empty:
        from ..analitica.huellas import PREFIJO_PILAR

        ids = {i.id: i.nombre for i in res.indicadores}
        L.append(f"HUELLAS DE LAS CAÍDAS ({len(hu.tramos)} tramos ≥{res.umbral_tramos:.0%} con indicadores, "
                 f"desde {hu.tramos['pico'].min():%Y})")
        lec = hu.lectura[~hu.lectura.index.str.startswith(PREFIJO_PILAR)]
        for papel in ("Anticipa: alto antes del pico", "Calma previa: bajo antes del pico", "Confirma: sube con la caída",
                      "Marca el piso: máximo en el valle"):
            d = lec[lec["papel"] == papel].sort_values("q_pre")
            if d.empty:
                continue
            L.append(f"  {papel} ({len(d)}):")
            for i, f in d.head(8).iterrows():
                L.append(f"    {ids.get(i, i):<40} 1 mes {f['media_pre21']:3.0f} · 1 sem {f['media_pre5']:3.0f} · "
                         f"conf {f['media_conf']:3.0f} · valle {f['media_valle']:3.0f} · normal {f['media_base']:3.0f}"
                         f" · zona alta 1 sem {f['alta_pre5']:.0%} vs {f['alta_base']:.0%} · q {f['q_pre']:.3f}")
        pl = hu.lectura[hu.lectura.index.str.startswith(PREFIJO_PILAR)]
        L.append("  Pilares (1 mes / 1 semana antes / pico / confirmación / valle · normal):")
        for i, f in pl.iterrows():
            c = i[len(PREFIJO_PILAR):]
            nombre = "Total" if c == "total" else PILARES.get(c, c)
            L.append(f"    {nombre:<28} {f['media_pre21']:3.0f} {f['media_pre5']:4.0f} {f['media_pico']:4.0f} "
                     f"{f['media_conf']:4.0f} {f['media_valle']:4.0f} · {f['media_base']:3.0f}  {f['papel']}")
        lc = hu.lectura_calma
        if not lc.empty:
            n_calma = int(hu.tramos["desde_calma"].sum())
            L.append(f"  Solo caídas desde la calma ({n_calma} tramos, 63+ ruedas después del valle anterior):")
            for i, f in lc[lc.index.str.startswith(PREFIJO_PILAR)].iterrows():
                c = i[len(PREFIJO_PILAR):]
                nombre = "Total" if c == "total" else PILARES.get(c, c)
                L.append(f"    {nombre:<28} {f['media_pre21']:3.0f} {f['media_pre5']:4.0f} {f['media_pico']:4.0f} "
                         f"{f['media_conf']:4.0f} {f['media_valle']:4.0f} · {f['media_base']:3.0f}  {f['papel']}")
            ant = lc[~lc.index.str.startswith(PREFIJO_PILAR) & lc["papel"].str.startswith(("Anticipa", "Calma"))]
            L.append("    Indicadores con señal antes del pico: " + ("; ".join(
                f"{ids.get(i, i)} ({f['media_pre5']:.0f} vs {f['media_base']:.0f}, {f['papel'].split(':')[0]})"
                for i, f in ant.sort_values("q_pre").iterrows()) or "ninguno significativo"))
        v = hu.vecinos_hoy
        L.append(f"  Análogos hoy: {_pct0(v.get('prob'))} de los {v.get('k')} días más parecidos antecedieron una "
                 f"caída ≥5 % en 21 ruedas (día cualquiera: {_pct0(hu.base_y3)})")
        ev = hu.evaluacion
        if not ev.empty:
            for _, f in ev[ev["periodo"] == "Todo el período"].iterrows():
                lift = f["tope20_y3"] / f["base_y3"] if f["base_y3"] else float("nan")
                L.append(f"    {f['variante']:<38} AUC 5d {f['auc_y1']:.3f} · 10d {f['auc_y2']:.3f} · 21d {f['auc_y3']:.3f}"
                         f" · top 20%: {f['tope20_y3']:.0%} ({lift:.2f}x)")
        if not hu.calibracion.empty:
            L.append("    Calibración: " + " · ".join(
                f"{k}: {f['realizado']:.0%} ({f['porcentaje_ruedas']:.0%} del tiempo)"
                for k, f in hu.calibracion.iterrows() if f["ruedas"]))
        if not hu.analogos_hoy.empty:
            L.append("  Días más parecidos a hoy: " + "; ".join(
                f"{d:%Y-%m-%d} (dist {f['distancia']:.0f}, peor caída 21r {f['caida_max_21']:+.1%})"
                for d, f in hu.analogos_hoy.iterrows()))
        if not hu.caidas_parecidas.empty:
            L.append("  Caídas con semana previa parecida: " + "; ".join(
                f"{f['pico']:%Y-%m-%d} {f['profundidad']:+.0%} {f.get('tipo', '')} (dist {f['distancia']:.0f})"
                for _, f in hu.caidas_parecidas.head(5).iterrows()))
        L.append("")

    if res.screener:
        L.append("SCREENER")
        for s in res.screener:
            if s["estado"] == "no_aplica":
                L.append(f"  {s['titulo']}: no aplica hoy")
            elif s["estado"] == "error":
                L.append(f"  {s['titulo']}: error en la consulta")
            else:
                t = s["resultado"]
                L.append(f"  {s['titulo']} ({len(t)}): " + (", ".join(t.index[:10]) if len(t) else "ninguno"))
        for k, v in res.canastas_corr.items():
            if "corr_63" in v and "corr_252" in v:
                L.append(f"  Correlación interna {k}: {v['corr_63']:.2f} (63r) vs {v['corr_252']:.2f} (252r)")
        L.append("")

    h = res.historia_larga
    if h is not None:
        tb = h.tabla.sort_values("puntaje_refugio", ascending=False)
        L.append(f"HISTORIA DESDE 1926 ({len(h.episodios)} mercados bajistas ≥20%)")
        L.append("  Mejores refugios: " + ", ".join(f"{i} ({v:.0f})" for i, v in tb["puntaje_refugio"].head(6).items()))
        L.append("  Peores:           " + ", ".join(f"{i} ({v:.0f})" for i, v in tb["puntaje_refugio"].tail(6).items()))
        L.append("")


    L.append(f"Registro forward: {len(res.registro_estado)} ruedas · opciones: {len(res.registro_opciones)} snapshots")
    return "\n".join(L)
