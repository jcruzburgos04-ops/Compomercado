"""Dashboard estático (HTML + Plotly) publicado en GitHub Pages."""

from __future__ import annotations

import html
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..indicadores.estado import PILARES
from ..pipeline import Resultados

log = logging.getLogger(__name__)

PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

# Paleta de referencia (skill dataviz): serie 1 azul, serie 2 naranja; divergente azul <-> rojo.
AZUL, NARANJA = "#2a78d6", "#eb6834"
DIVERGENTE = [
    [0.0, "#a8322f"], [0.2, "#e34948"], [0.4, "#f3b3b2"], [0.5, "#f0efec"],
    [0.6, "#b7d3f6"], [0.8, "#3987e5"], [1.0, "#184f95"],
]
TINTA, TINTA_2, MUTED, GRILLA, EJE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FUENTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'

NIVELES = [  # (hasta, etiqueta, clase, icono) — umbrales preliminares, se calibran en la Fase 3
    (40, "Bajo", "good", "✓"),
    (60, "Moderado", "warning", "▲"),
    (75, "Elevado", "serious", "▲▲"),
    (101, "Alto", "critical", "✖"),
]


# --- formato ---------------------------------------------------------------------------------

def _nulo(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v)) or (isinstance(v, float) and math.isinf(v))
    except (TypeError, ValueError):
        return False


def fmt(v, tipo: str = "num") -> str:
    if _nulo(v):
        return "—"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if tipo == "pct":
        return f"{v * 100:+.1f} %" if abs(v) < 10 else f"{v * 100:+.0f} %"
    if tipo == "pct0":
        return f"{v * 100:.0f} %"
    if tipo == "x":
        return f"{v:.2f}x"
    if tipo == "int":
        return f"{v:.0f}"
    if tipo == "puntaje":
        return f"{v:.0f}"
    if tipo == "delta":
        return f"{v:+.0f}"
    if tipo == "texto":
        return html.escape(str(v))
    if tipo == "fecha":
        return pd.Timestamp(v).strftime("%Y-%m-%d") if pd.notna(v) else "—"
    return f"{v:.2f}"


def _valor_orden(v) -> str:
    if _nulo(v):
        return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.Timestamp(v).strftime("%Y%m%d")
    if isinstance(v, (int, float, np.floating, np.integer)):
        return f"{float(v):.6g}"
    return html.escape(str(v))


def nivel(v: float) -> tuple[str, str, str]:
    for hasta, etiqueta, clase, icono in NIVELES:
        if v < hasta:
            return etiqueta, clase, icono
    return NIVELES[-1][1:]


def tabla(df: pd.DataFrame, columnas: list[tuple], id_: str = "", atributos_fila=None, clase: str = "") -> str:
    """columnas: (col, título, tipo[, ayuda]). tipo 'puntaje' dibuja una barra 0-100."""
    cab = []
    for c in columnas:
        ayuda = f' title="{html.escape(c[3])}"' if len(c) > 3 else ""
        num = "" if c[2] in ("texto", "fecha") else ' class="n"'
        cab.append(f"<th{num}{ayuda}>{html.escape(c[1])}</th>")
    filas = []
    for idx, fila in df.iterrows():
        attrs = atributos_fila(idx, fila) if atributos_fila else ""
        celdas = []
        for col, _, tipo, *_ in columnas:
            v = idx if col == "__indice__" else fila.get(col)
            if _nulo(v):
                v = None
            num = "" if tipo in ("texto", "fecha") else ' class="n"'
            if tipo == "puntaje" and v is not None:
                contenido = f'<span class="barra" style="--v:{max(0, min(100, v)):.0f}%"></span>{fmt(v, tipo)}'
            else:
                contenido = fmt(v, tipo)
            celdas.append(f'<td{num} data-v="{_valor_orden(v)}">{contenido}</td>')
        filas.append(f"<tr{attrs}>{''.join(celdas)}</tr>")
    id_attr = f' id="{id_}"' if id_ else ""
    return (f'<div class="tabla-envoltura"><table class="t ordenable {clase}"{id_attr}><thead><tr>'
            f'{"".join(cab)}</tr></thead><tbody>{"".join(filas)}</tbody></table></div>')


# --- gráficos --------------------------------------------------------------------------------

def _layout(fig: go.Figure, alto: int = 360, leyenda: bool = False) -> go.Figure:
    fig.update_layout(
        height=alto,
        margin=dict(l=48, r=16, t=28, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FUENTE, size=12, color=TINTA_2),
        hovermode="x unified",
        showlegend=leyenda,
        legend=dict(orientation="h", y=1.08, x=0),
        hoverlabel=dict(font=dict(family=FUENTE)),
    )
    fig.update_xaxes(showgrid=False, linecolor=EJE, ticks="outside", tickcolor=EJE)
    fig.update_yaxes(gridcolor=GRILLA, zeroline=False, linecolor=EJE)
    return fig


def grafico(fig: go.Figure, id_: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=id_,
                       config={"displaylogo": False, "responsive": True,
                               "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def _selector_rango(fig: go.Figure) -> None:
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(count=3, label="3a", step="year", stepmode="backward"),
                dict(count=10, label="10a", step="year", stepmode="backward"),
                dict(step="all", label="Todo"),
            ],
            bgcolor="rgba(0,0,0,0)", activecolor="rgba(42,120,214,0.18)",
        ),
        row=1, col=1,
    )


def _heatmap(z: pd.DataFrame, titulo_color: str, zmax: float, formato: str = ".1%", alto=None) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=z.to_numpy(dtype=float), x=list(z.columns), y=list(z.index),
            colorscale=DIVERGENTE, zmid=0, zmin=-zmax, zmax=zmax, xgap=2, ygap=2,
            colorbar=dict(title=titulo_color, thickness=10, tickformat=formato),
            hovertemplate="%{y}<br>%{x}<br>" + titulo_color + ": %{z:" + formato + "}<extra></extra>",
            meta="divergente",
        )
    )
    fig = _layout(fig, alto or max(320, 22 * len(z) + 120))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


# --- secciones -------------------------------------------------------------------------------

def seccion_estado(res: Resultados) -> str:
    pil = res.pilares
    if pil.empty or pil["total"].dropna().empty:
        return "<p>Sin datos suficientes para el panel de estado.</p>"
    total = pil["total"].dropna()
    ult, previo = total.iloc[-1], total.iloc[-6] if len(total) > 5 else np.nan
    etiqueta, clase, icono = nivel(ult)

    pond = res.ponderacion
    vig = pond.pesos_vigentes if pond is not None else None
    peso_ind = vig.indicadores["peso"] if vig is not None else pd.Series(dtype=float)
    estado_ind = vig.indicadores["estado"] if vig is not None else pd.Series(dtype=object)

    # Tabla de pilares con el indicador que más empuja (entre los que cuentan para el pilar).
    ultimos = res.riesgo.ffill().iloc[-1]
    filas = []
    for pilar, nombre in PILARES.items():
        if pilar not in pil.columns:
            continue
        s = pil[pilar].dropna()
        ids = [i for i in res.indicadores if i.pilar == pilar and i.id in ultimos.index and pd.notna(ultimos[i.id])
               and (vig is None or peso_ind.get(i.id, 0) > 0)]
        motor = max(ids, key=lambda i: ultimos[i.id], default=None)
        calma = min(ids, key=lambda i: ultimos[i.id], default=None)
        filas.append({
            "pilar": nombre,
            "puntaje": s.iloc[-1] if len(s) else np.nan,
            "delta": s.iloc[-1] - s.iloc[-6] if len(s) > 5 else np.nan,
            "empuja": f"{motor.nombre} (p{ultimos[motor.id]:.0f})" if motor else "",
            "calma": f"{calma.nombre} (p{ultimos[calma.id]:.0f})" if calma and calma is not motor else "",
            "peso": vig.pilares.loc[pilar, "peso"] if vig is not None and pilar in vig.pilares.index else np.nan,
            "activos": (f"{int(vig.pilares.loc[pilar, 'incluidos'])} de {int(vig.pilares.loc[pilar, 'total'])}"
                        if vig is not None and pilar in vig.pilares.index else ""),
        })
    cols_pil = [("pilar", "Pilar", "texto"), ("puntaje", "Puntaje", "puntaje"), ("delta", "Δ 5 ruedas", "delta")]
    if vig is not None:
        cols_pil += [("peso", "Peso en el total", "pct0", "Peso por ranking de importancia (ver Ponderaciones)"),
                     ("activos", "Indicadores que cuentan", "texto")]
    cols_pil += [("empuja", "Lo que más empuja", "texto"), ("calma", "Lo que más calma", "texto")]
    t_pilares = tabla(pd.DataFrame(filas), cols_pil)

    # Tabla de indicadores.
    filas = []
    for ind in res.indicadores:
        s = ind.serie.dropna()
        if s.empty:
            continue
        filas.append({
            "pilar": PILARES.get(ind.pilar, ind.pilar), "nombre": ind.nombre,
            "valor": s.iloc[-1], "formato": ind.formato,
            "delta": s.iloc[-1] - s.iloc[-6] if len(s) > 5 else np.nan,
            "riesgo": res.riesgo[ind.id].dropna().iloc[-1] if ind.id in res.riesgo and res.riesgo[ind.id].notna().any() else np.nan,
            "fecha": s.index[-1], "desc": ind.descripcion or ("Solo contexto" if not ind.en_compuesto else ""),
            "signo": {1: "↑ más = más riesgo", -1: "↓ menos = más riesgo", 0: "contexto"}[ind.signo],
            "peso": peso_ind.get(ind.id, np.nan) if ind.en_compuesto else np.nan,
            "estado": estado_ind.get(ind.id, "solo contexto" if not ind.en_compuesto else ""),
        })
    df_ind = pd.DataFrame(filas)
    df_ind["valor_txt"] = [fmt(v, f) for v, f in zip(df_ind["valor"], df_ind["formato"], strict=True)]
    df_ind["delta_txt"] = [fmt(v, f) for v, f in zip(df_ind["delta"], df_ind["formato"], strict=True)]
    t_ind = tabla(df_ind, [
        ("pilar", "Pilar", "texto"), ("nombre", "Indicador", "texto"), ("valor_txt", "Valor", "texto"),
        ("delta_txt", "Δ 5 ruedas", "texto"), ("riesgo", "Percentil de riesgo", "puntaje"),
        ("signo", "Lectura", "texto"),
    ] + ([("peso", "Peso en su pilar", "pct0"), ("estado", "Estado", "texto")] if vig is not None else []) + [
        ("fecha", "Último dato", "fecha"), ("desc", "Nota", "texto"),
    ])

    # Gráfico: puntaje total y SPY (dos paneles, un eje cada uno).
    spy = res.precios["SPY"].dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.45, 0.55],
                        subplot_titles=("Puntaje de riesgo total (0-100)", "SPY (escala log)"))
    fig.add_trace(go.Scatter(x=total.index, y=total, name="Puntaje", line=dict(color=AZUL, width=2),
                             hovertemplate="%{y:.0f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=spy.index, y=spy, name="SPY", line=dict(color=AZUL, width=2),
                             hovertemplate="%{y:.2f}"), row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig = _layout(fig, 560)
    _selector_rango(fig)
    inicio = total.index[-1] - pd.DateOffset(years=5)
    fig.update_xaxes(range=[inicio, total.index[-1]])

    # Pilares como pequeños múltiplos.
    pilares_ok = [p for p in PILARES if p in pil.columns]
    filas_n = math.ceil(len(pilares_ok) / 2)
    fig_p = make_subplots(rows=filas_n, cols=2, shared_xaxes=True, vertical_spacing=0.08,
                          subplot_titles=[PILARES[p] for p in pilares_ok])
    desde = pil.index[-1] - pd.DateOffset(years=3)
    for k, p in enumerate(pilares_ok):
        s = pil[p].loc[desde:]
        fig_p.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=AZUL, width=1.5), name=PILARES[p],
                                   hovertemplate="%{y:.0f}"), row=k // 2 + 1, col=k % 2 + 1)
    fig_p.update_yaxes(range=[0, 100])
    fig_p = _layout(fig_p, 170 * filas_n + 60)

    base = res.tabla_base.copy()
    t_base = tabla(base, [
        ("__indice__", "Puntaje total", "texto"), ("ruedas", "Ruedas", "int"),
        ("caida3_5d", "Caída ≥3 % en 5 ruedas", "pct0"), ("caida5_10d", "Caída ≥5 % en 10 ruedas", "pct0"),
        ("caida5_21d", "Caída ≥5 % en 21 ruedas", "pct0"), ("ret21_medio", "Retorno medio 21 ruedas", "pct"),
    ])

    dias = len(total)
    igual = res.pilares_igual["total"].dropna() if "total" in res.pilares_igual else pd.Series(dtype=float)
    if vig is not None:
        etiqueta_total = "Riesgo total (ponderado)"
        comparacion = f" · con pesos iguales: {fmt(igual.iloc[-1], 'puntaje')}" if len(igual) else ""
        como = (f"Cada indicador se compara con toda su historia previa (percentil point-in-time) y se orienta "
                f"para que 100 = más riesgo. Dentro de cada pilar y entre pilares, <strong>pesa más lo que más "
                f"anticipó caídas</strong>: los pesos se recalculan cada enero solo con datos anteriores (ver la "
                f"pestaña Ponderaciones). <strong>No es una probabilidad.</strong> La tabla de abajo muestra qué "
                f"pasó después de cada nivel desde {total.index[0]:%Y} ({dias:,} ruedas, fuera de muestra para los pesos).")
    else:
        etiqueta_total, comparacion = "Riesgo total (preliminar)", ""
        como = (f"Cada indicador se compara con toda su historia previa (percentil point-in-time) y se orienta "
                f"para que 100 = más riesgo. Cada pilar promedia sus indicadores y el total promedia los pilares. "
                f"<strong>No es una probabilidad.</strong> La tabla de abajo muestra qué pasó después de cada "
                f"nivel ({dias:,} ruedas; descriptivo, dentro de la muestra).")
    return f"""
<div class="hero">
  <div class="tile">
    <div class="tile-etq">{etiqueta_total}</div>
    <div class="tile-num">{ult:.0f}<span class="tile-de">/100</span></div>
    <div class="estado {clase}"><span aria-hidden="true">{icono}</span> {etiqueta}</div>
    <div class="tile-sub">Hace 5 ruedas: {fmt(previo, 'puntaje')}{comparacion} · datos al {res.fecha:%d/%m/%Y}</div>
  </div>
  <div class="nota">
    <p><strong>Cómo leerlo.</strong> {como}</p>
  </div>
</div>
<h3>Pilares</h3>{t_pilares}
<h3>Qué pasó después, según el nivel del puntaje</h3>{t_base}
<p class="pie">Base: todas las ruedas con puntaje. Los objetivos son los de la metodología (horizonte swing).
Los umbrales del semáforo (40 / 60 / 75) siguen siendo provisorios.</p>
{grafico(fig, "g-total")}
<h3>Pilares, últimos 3 años</h3>{grafico(fig_p, "g-pilares")}
<h3>Indicadores</h3>{t_ind}
"""


def seccion_ponderaciones(res: Resultados) -> str:
    pond = res.ponderacion
    if pond is None:
        return "<p>Todavía no hay historia suficiente para estimar pesos.</p>"
    vig = pond.pesos_vigentes
    ev = pond.evaluacion.copy()

    # Evaluación fuera de muestra.
    partes_ev = []
    for per in ev["periodo"].unique():
        t = ev[ev["periodo"] == per].set_index("variante")
        t["lift"] = t["tope20_y3"] / t["base_y3"]
        cols = [("__indice__", "Variante", "texto"), ("ruedas", "Ruedas", "int"),
                ("auc_y1", "AUC 3 % / 5 ruedas", "num", "0,5 = no anticipa; 1 = separa perfecto"),
                ("auc_y2", "AUC 5 % / 10 ruedas", "num"), ("auc_y3", "AUC 5 % / 21 ruedas", "num"),
                ("tope20_y3", "Caída ≥5 %/21r en el 20 % más riesgoso", "pct0"),
                ("base_y3", "Caída ≥5 %/21r en todos los días", "pct0"),
                ("lift", "Cuántas veces más", "x", "Frecuencia en el 20 % más riesgoso / frecuencia base")]
        partes_ev.append(f"<h4>{html.escape(per)}</h4>{tabla(t, cols)}")

    # Pilares y pesos.
    tp = vig.pilares.copy()
    tp.index = [PILARES.get(p, p) for p in tp.index]
    tp["activos"] = [f"{int(a)} de {int(b)}" for a, b in zip(tp["incluidos"], tp["total"], strict=True)]
    t_pil = tabla(tp.sort_values("peso", ascending=False), [
        ("__indice__", "Pilar", "texto"), ("peso", "Peso en el total", "pct0"),
        ("auc_media", "Importancia (AUC medio)", "num"), ("activos", "Indicadores que cuentan", "texto"),
        ("estado", "Estado", "texto")])

    # Indicadores por pilar.
    nombres = {i.id: i.nombre for i in res.indicadores}
    ti = vig.indicadores.copy()
    ti["nombre"] = [nombres.get(i, i) for i in ti.index]
    ti["pilar_n"] = [PILARES.get(p, p) for p in ti["pilar"]]
    orden_pil = {p: k for k, p in enumerate(PILARES)}
    ti["_o"] = [orden_pil.get(p, 99) for p in ti["pilar"]]
    ti = ti.sort_values(["_o", "peso", "auc_media"], ascending=[True, False, False])
    t_ind = tabla(ti, [
        ("pilar_n", "Pilar", "texto"), ("nombre", "Indicador", "texto"), ("peso", "Peso en su pilar", "pct0"),
        ("auc_media", "AUC medio", "num"), ("auc_y1", "AUC 3 %/5r", "num"), ("auc_y2", "AUC 5 %/10r", "num"),
        ("auc_y3", "AUC 5 %/21r", "num"), ("n", "Ruedas evaluadas", "int"), ("estado", "Estado", "texto")])

    # Historial de pesos por pilar (walk-forward).
    h = pond.historial.copy()
    h.columns = [PILARES.get(c, c) for c in h.columns]
    h.index = [str(a) for a in h.index]
    t_hist = tabla(h.sort_index(ascending=False), [("__indice__", "Año", "texto")] + [(c, c, "pct0") for c in h.columns])

    return f"""
<p>Los pesos salen de cuánto anticipó cada indicador las caídas de SPY en el horizonte swing (3 % en 5 ruedas,
5 % en 10 y 5 % en 21). La medida es el AUC: 0,5 = no anticipa nada, 1 = separa perfecto los días previos a una
caída del resto. Con eso:</p>
<ul>
<li><strong>Se depura:</strong> queda afuera el indicador con poca historia, el que no anticipa (AUC &lt; 0,52), el que
funciona al revés de lo esperado (AUC &lt; 0,48; no se le da vuelta el sentido, porque sería forzar los datos) y el
que repite a otro más importante del mismo pilar (correlación ≥ 0,85).</li>
<li><strong>Se pondera por ranking:</strong> el más importante del pilar pesa n, el siguiente n−1, … el último 1.
Lo mismo entre pilares.</li>
<li><strong>Sin mirar el futuro:</strong> los pesos de cada año se calculan solo con datos hasta 21 ruedas antes del 1
de enero. Los vigentes se estimaron con datos hasta el {vig.hasta:%d/%m/%Y}.</li>
</ul>
<h3>¿Ponderar mejora? Evaluación fuera de muestra</h3>
<p class="pie">Cada variante se mide en años que no participaron del cálculo de sus pesos. Las referencias simples
(SPY bajo su media de 200, nivel del VIX) no tienen pesos: son la vara mínima a superar.</p>
{"".join(partes_ev)}
<h3>Peso de cada pilar (vigente)</h3>{t_pil}
<h3>Peso e importancia de cada indicador (vigente)</h3>{t_ind}
<h3>Peso de cada pilar, año por año</h3>
<p class="pie">Si los pesos saltan mucho de un año a otro, la importancia es inestable y conviene no confiar en ella.</p>
{t_hist}
"""


COLS_MAPA = [
    ("__indice__", "Activo", "texto"), ("nombre", "Nombre", "texto"), ("grupo", "Grupo", "texto"),
    ("puntaje_refugio", "Refugio", "puntaje", "Qué tan bien aguanta las caídas (0-100 dentro del universo)"),
    ("puntaje_rebote", "Rebote", "puntaje", "Qué tan bien lidera la salida de las caídas"),
    ("rs_63_pct", "Fuerza rel. 63d", "puntaje", "Percentil del retorno relativo de 63 ruedas"),
    ("beta", "Beta", "num"), ("beta_bajista", "Beta ↓", "num", "Beta en días de caída del mercado"),
    ("beta_alcista", "Beta ↑", "num"), ("captura_bajista", "Captura ↓", "num", "Morningstar, meses de caída"),
    ("captura_alcista", "Captura ↑", "num"), ("corr_estres", "Correl. estrés", "num", "Correlación en el peor 10 % de días"),
    ("beta_cola", "Beta cola", "num", "Retorno medio en el peor 5 % de días del mercado / el del mercado"),
    ("n_episodios", "Episodios", "int"), ("captura_mediana", "Captura mediana", "num", "Mediana de (ret. activo / ret. mercado) por episodio"),
    ("acierto_defensivo", "Acierto defensivo", "pct0", "% de episodios en que le ganó al mercado"),
    ("lag_techo_mediano", "Techo (ruedas)", "int", "Negativo = hace techo antes que el mercado"),
    ("lag_piso_mediano", "Piso (ruedas)", "int", "Negativo = toca fondo antes que el mercado"),
    ("rebote_rel_63_mediano", "Rebote rel. 63d", "pct"), ("frecuencia", "Datos", "texto"),
    ("inicio", "Desde", "fecha"),
]


def _mapa_tabla(t: pd.DataFrame, res: Resultados, id_: str) -> str:
    t = t.copy()
    t["nombre"] = [res.nombres.get(i, i) for i in t.index]
    t = t.sort_values("puntaje_refugio", ascending=False)
    attrs = lambda idx, fila: f' data-grupo="{html.escape(str(fila.get("grupo", "")))}"'  # noqa: E731
    return tabla(t, COLS_MAPA, id_, attrs, clase="filtrable")


def seccion_mapa(res: Resultados) -> str:
    if not res.mapas:
        return "<p>Sin mapa.</p>"
    grupos = sorted({g for m in res.mapas.values() for g in m.tabla["grupo"].dropna().unique() if g})
    opciones = "".join(f'<option value="{g}">{g}</option>' for g in grupos)
    pestañas, paneles = [], []
    for k, (ref, m) in enumerate(res.mapas.items()):
        activo = " activo" if k == 0 else ""
        pestañas.append(f'<button class="subtab{activo}" data-panel="mapa-{k}">vs {ref}</button>')
        n_eps = len(m.episodios)
        paneles.append(f'<div class="subpanel{activo}" id="mapa-{k}"><p class="pie">{n_eps} tramos de caída ≥10 % '
                       f'de {ref} desde {m.episodios["pico"].min():%Y} (zigzag).</p>{_mapa_tabla(m.tabla, res, f"t-mapa-{k}")}</div>')

    spy = res.mapas.get("SPY")
    extras = ""
    if spy is not None:
        t = spy.tabla
        # Solo instrumentos operables y medidos con datos diarios (sin índices "^").
        mask_sinc = (t["frecuencia"] == "diaria") & ~t.index.str.startswith("^")
        acciones = t["grupo"].isin(["sectores", "industrias", "factores", "global_etf", "referencias", "canasta"])
        fuera = t["grupo"].isin(["renta_fija", "commodities", "divisas", "cripto"])
        # Refugio y fuerza relativa se rankean dentro de cada clase: comparar bonos con acciones
        # llena la lista de bonos, y para rotar dentro de la renta variable eso no sirve.
        def rango_clase(mask):
            d = t[mask_sinc & mask].copy()
            d["ref_clase"] = d["puntaje_refugio"].rank(pct=True) * 100
            d["rs_clase"] = d["rs_63"].rank(pct=True) * 100 if "rs_63" in d else np.nan
            d["nombre"] = [res.nombres.get(i, i) for i in d.index]
            return d

        acc, otros = rango_clase(acciones), rango_clase(fuera)
        ref_acc = acc[(acc["ref_clase"] >= 60) & (acc["rs_clase"] >= 50)].sort_values("ref_clase", ascending=False).head(12)
        ref_otros = otros[(otros["ref_clase"] >= 50)].sort_values("rs_clase", ascending=False).head(8)
        umbral_beta = acc["beta_bajista"].quantile(0.75)
        reducir = acc[(acc["beta_bajista"] >= umbral_beta) & (acc["rs_clase"] < 50)].sort_values("beta_bajista", ascending=False).head(12)
        cols_r = [("__indice__", "Activo", "texto"), ("nombre", "Nombre", "texto"),
                  ("ref_clase", "Refugio (en su clase)", "puntaje"), ("rs_clase", "Fuerza rel. 63d (en su clase)", "puntaje"),
                  ("captura_mediana", "Captura mediana", "num"), ("acierto_defensivo", "Acierto", "pct0")]
        cols_d = [("__indice__", "Activo", "texto"), ("nombre", "Nombre", "texto"), ("beta_bajista", "Beta ↓", "num"),
                  ("rs_clase", "Fuerza rel. 63d (en su clase)", "puntaje"), ("captura_mediana", "Captura mediana", "num")]
        extras += f"""
<div class="dos">
  <div><h3>Rotar dentro de acciones: refugios con fuerza hoy</h3><p class="pie">Sectores, industrias, factores y
  países en el 40 % superior de refugio de las acciones, con fuerza relativa de 63 ruedas en la mitad superior.</p>{tabla(ref_acc, cols_r)}</div>
  <div><h3>Reducir primero si sube el riesgo</h3><p class="pie">Acciones con beta bajista en el cuartil superior y
  fuerza relativa débil. Los líderes con fuerza no aparecen acá.</p>{tabla(reducir, cols_d)}</div>
</div>
<h3>Refugios fuera de acciones</h3><p class="pie">Bonos, oro, commodities y divisas con buen historial en las caídas,
ordenados por fuerza relativa actual. Mirar la correlación acciones-bonos (pestaña Correlaciones): si es positiva,
los bonos cubren menos.</p>{tabla(ref_otros, cols_r)}
"""
        if not spy.por_tipo.empty:
            pt = spy.por_tipo.copy()
            pt = pt[pt.index.isin(t.index[t["grupo"].isin(["sectores", "factores", "renta_fija", "commodities", "industrias"])])]
            pt = pt.loc[t.loc[pt.index, "puntaje_refugio"].sort_values(ascending=False).index]
            pt.insert(0, "nombre", [res.nombres.get(i, i) for i in pt.index])
            cols = [("__indice__", "Activo", "texto"), ("nombre", "Nombre", "texto")] + [(c, c, "pct") for c in spy.por_tipo.columns]
            extras += f"""<h3>Retorno relativo mediano según el tipo de caída</h3>
<p class="pie">Tipos asignados por reglas simples sobre la huella macro de cada tramo (ver pestaña Caídas).
Son pocos episodios por tipo: leer como orientación, no como regla.</p>{tabla(pt, cols)}"""

        grupos_hm = ["sectores", "industrias", "factores", "renta_fija", "commodities"]
        filas = t.index[t["grupo"].isin(grupos_hm) & mask_sinc]
        piv = spy.larga[spy.larga["activo"].isin(filas)].pivot_table(index="activo", columns="episodio", values="relativo")
        if not piv.empty:
            eps = spy.episodios
            piv.columns = [f"{eps.loc[c, 'pico']:%Y-%m} ({eps.loc[c, 'profundidad']:.0%})" for c in piv.columns]
            orden = t.loc[piv.index, "puntaje_refugio"].sort_values(ascending=False).index
            piv = piv.loc[orden]
            piv.index = [f"{i} · {res.nombres.get(i, i)}" for i in piv.index]
            extras += ('<h3>Retorno relativo contra SPY en cada tramo de caída</h3><p class="pie">Azul = cayó menos '
                       'que SPY; rojo = cayó más. Ordenado por puntaje de refugio.</p>'
                       + grafico(_heatmap(piv.clip(-0.3, 0.3), "Relativo", 0.2), "g-hm-mapa"))

    return f"""
<p>Cómo se comporta cada activo frente a cada referencia: en general (betas, captura, correlación en estrés)
y en cada tramo de caída (retorno relativo, quién hace techo y piso primero, quién lidera el rebote).
Los activos que no cotizan en horario de EE. UU. se miden con retornos semanales.</p>
{extras}
<h3>Tabla completa</h3>
<div class="filtros"><div class="subtabs">{"".join(pestañas)}</div>
<label>Grupo <select class="filtro-grupo"><option value="">Todos</option>{opciones}</select></label></div>
{"".join(paneles)}
<p class="pie">Clic en un encabezado para ordenar. Pasá el mouse sobre los encabezados para ver cada definición.</p>
"""


def seccion_episodios(res: Resultados) -> str:
    spy = res.mapas.get("SPY")
    if spy is None:
        return "<p>Sin episodios.</p>"
    eps = spy.episodios.copy()
    p = res.precios["SPY"].dropna()
    fig = go.Figure(go.Scatter(x=p.index, y=p, line=dict(color=AZUL, width=1.5), name="SPY", hovertemplate="%{y:.2f}"))
    for _, e in eps.iterrows():
        fig.add_vrect(x0=e["pico"], x1=e["valle"], fillcolor="rgba(227,73,72,0.14)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=eps["valle"], y=eps["precio_valle"], mode="markers", marker=dict(size=8, color=NARANJA, line=dict(width=2, color="#fcfcfb")),
        name="Valle", customdata=np.stack([eps["profundidad"], eps["dias_caida"], eps.get("tipo", pd.Series("", index=eps.index))], axis=1),
        hovertemplate="Valle %{x|%Y-%m-%d}<br>Caída %{customdata[0]:.1%} en %{customdata[1]:.0f} ruedas<br>%{customdata[2]}<extra></extra>",
    ))
    fig.update_yaxes(type="log")
    fig = _layout(fig, 420)
    fig.update_layout(hovermode="closest")

    cols = [("pico", "Pico", "fecha"), ("valle", "Valle", "fecha"), ("profundidad", "Caída", "pct"),
            ("dias_caida", "Ruedas de caída", "int"), ("dias_recuperacion", "Ruedas hasta recuperar", "int"),
            ("tipo", "Tipo (tentativo)", "texto"), ("d_dgs10", "Δ tasa 10a (pp)", "num"),
            ("d_baa10y", "Δ spread Baa (pp)", "num"), ("r_dxy", "Dólar", "pct"), ("r_wti", "Petróleo", "pct"),
            ("r_usdjpy", "USDJPY", "pct"), ("r_oro", "Oro", "pct"), ("vix_max", "VIX máx.", "num")]
    cols = [c for c in cols if c[0] in eps.columns]
    ba = res.bajo_agua
    cols_ba = [("pico", "Pico", "fecha"), ("valle", "Valle", "fecha"), ("recuperacion", "Recuperó", "fecha"),
               ("profundidad", "Caída", "pct"), ("dias_caida", "Ruedas de caída", "int"),
               ("dias_recuperacion", "Ruedas hasta recuperar", "int")]
    return f"""
<p>Cada tramo de caída ≥10 % de SPY (retorno total) desde 1993, con lo que hicieron tasas, crédito, dólar,
petróleo y yen entre el pico y el valle. Un tramo termina cuando SPY rebota ≥10 % desde el valle, así que una
caída larga (2008, 2022) aparece como varias piernas.</p>
{grafico(fig, "g-episodios")}
<h3>Tramos de caída ≥10 % y su huella macro</h3>{tabla(eps.sort_values("pico", ascending=False), cols)}
<h3>Episodios completos (pico → valle → recuperación del pico), ≥10 %</h3>{tabla(ba.sort_values("pico", ascending=False), cols_ba)}
"""


def seccion_historia(res: Resultados) -> str:
    h = res.historia_larga
    if h is None:
        return "<p>No se pudo descargar la base de Kenneth French.</p>"
    t = h.tabla.copy().sort_values("puntaje_refugio", ascending=False)
    cols = [("__indice__", "Industria", "texto"), ("puntaje_refugio", "Refugio", "puntaje"), ("puntaje_rebote", "Rebote", "puntaje"),
            ("beta", "Beta", "num"), ("beta_bajista", "Beta ↓", "num"), ("captura_bajista", "Captura ↓", "num"),
            ("corr_estres", "Correl. estrés", "num"), ("n_episodios", "Episodios", "int"),
            ("captura_mediana", "Captura mediana", "num"), ("acierto_defensivo", "Acierto defensivo", "pct0"),
            ("lag_piso_mediano", "Piso (ruedas)", "int"), ("rebote_rel_63_mediano", "Rebote rel. 63d", "pct"),
            ("inicio", "Desde", "fecha")]
    piv = h.larga.pivot_table(index="activo", columns="episodio", values="relativo")
    eps = h.episodios
    piv.columns = [f"{eps.loc[c, 'pico']:%Y-%m} ({eps.loc[c, 'profundidad']:.0%})" for c in piv.columns]
    piv = piv.loc[[i for i in t.index if i in piv.index]]
    return f"""
<p>Las 49 industrias de Kenneth French (retornos diarios desde julio de 1926, sin sesgo de supervivencia) en
cada mercado bajista ≥20 % del mercado de EE. UU. Sirve para ver si los patrones de la era de los ETF
(desde 1998) se sostienen en 1929, 1937, 1973-74, 1987 y el resto.</p>
{tabla(t, cols)}
<h3>Retorno relativo contra el mercado en cada mercado bajista</h3>
{grafico(_heatmap(piv.clip(-0.4, 0.4), "Relativo", 0.3), "g-hm-ff")}
"""


def seccion_correlacion(res: Resultados) -> str:
    c = res.correlacion
    fig = _heatmap(c.round(2), "Correlación", 1.0, ".2f", alto=640)
    fig.update_layout(margin=dict(l=60, r=16, t=16, b=60))
    ids = {i.id: i for i in res.indicadores}
    series = [(k, ids[k]) for k in ("corr_prom", "absorcion", "turbulencia", "corr_acc_bonos") if k in ids]
    fig2 = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=[i.nombre for _, i in series])
    for k, (_, ind) in enumerate(series):
        s = ind.serie.dropna()
        fig2.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=AZUL, width=1.5), name=ind.nombre,
                                  hovertemplate="%{y:.2f}"), row=k + 1, col=1)
    fig2 = _layout(fig2, 180 * len(series) + 60)
    return f"""
<p>Correlación de los últimos 63 días entre índices, sectores y activos refugio. Cuando todo se mueve junto
(rojo generalizado) la diversificación desaparece: suele pasar antes y durante las caídas.</p>
{grafico(fig, "g-corr")}
<h3>Estructura de correlación en el tiempo</h3>{grafico(fig2, "g-corr-t")}
"""


def seccion_argentina(res: Resultados) -> str:
    a = res.argentina
    if not a:
        return "<p>Sin datos de Argentina.</p>"
    series = []
    if "ccl" in a and not a["ccl"].dropna().empty:
        series.append(("CCL implícito (ARS por USD)", a["ccl"].dropna(), True))
    if "merval_usd" in a:
        series.append(("Merval en dólares CCL", a["merval_usd"].dropna(), True))
    if "exposicion" in a:
        series.append(("R² contra SPY, EEM y commodities (252 ruedas)", a["exposicion"]["r2_global"].dropna(), False))
        series.append(("Residuo local acumulado 63 ruedas", a["exposicion"]["residuo_63"].dropna(), False))
    partes = []
    if series:
        fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=[s[0] for s in series])
        for k, (nombre, s, log_) in enumerate(series):
            fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=AZUL, width=1.5), name=nombre,
                                     hovertemplate="%{y:.2f}"), row=k + 1, col=1)
            if log_:
                fig.update_yaxes(type="log", row=k + 1, col=1)
        fig = _layout(fig, 200 * len(series) + 60)
        _selector_rango(fig)
        partes.append(grafico(fig, "g-arg"))
    cols = [("__indice__", "Activo", "texto"), ("nombre", "Nombre", "texto"), ("puntaje_refugio", "Refugio", "puntaje"),
            ("beta", "Beta", "num"), ("beta_bajista", "Beta ↓", "num"), ("captura_bajista", "Captura ↓", "num"),
            ("corr_estres", "Correl. estrés", "num"), ("n_episodios", "Episodios", "int"),
            ("captura_mediana", "Captura mediana", "num"), ("acierto_defensivo", "Acierto", "pct0"),
            ("rs_63_pct", "Fuerza rel. 63d", "puntaje"), ("inicio", "Desde", "fecha")]
    for clave, titulo in (("mapa_spy", "ADRs frente a las caídas de SPY"), ("mapa_eem", "ADRs frente a las caídas de emergentes (EEM)")):
        if clave in a:
            t = a[clave].copy()
            t["nombre"] = [res.nombres.get(i, i) for i in t.index]
            partes.append(f"<h3>{titulo}</h3>{tabla(t.sort_values('beta_bajista'), cols)}")
    ultimo = ""
    if "exposicion" in a and not a["exposicion"]["r2_global"].dropna().empty:
        r2 = a["exposicion"]["r2_global"].dropna().iloc[-1]
        lectura = ("el mercado global explica buena parte del movimiento: las señales del sensor aplican"
                   if r2 >= 0.35 else "domina el factor local/político: las señales globales pesan poco hoy")
        ultimo = f'<p class="destacado">R² actual: <strong>{r2:.0%}</strong> — {lectura}.</p>'
    return f"""
<p>Argentina se mide como un activo expuesto al estrés global. El CCL se calcula con la mediana de varias
especies (precio local × acciones por ADR / precio del ADR), descartando las que se alejan más de 15 %.
La regresión móvil separa cuánto del movimiento de la canasta de ADRs explica el mundo y cuánto es local.</p>
{ultimo}{"".join(partes)}
"""


def seccion_forward(res: Resultados) -> str:
    r = res.registro_estado
    if r.empty:
        cuerpo = "<p>El registro arranca con la primera corrida diaria en GitHub Actions.</p>"
    else:
        cols = [("__indice__", "Fecha", "fecha"), ("registrado_utc", "Registrado (UTC)", "texto")]
        if "pond_total" in r.columns:
            cols.append(("pond_total", "Total ponderado", "puntaje"))
        cols.append(("pilar_total", "Total pesos iguales", "puntaje"))
        # Las columnas de pilares se muestran solo si existen en el registro.
        cols += [(f"pilar_{p}", n, "puntaje") for p, n in PILARES.items() if f"pilar_{p}" in r.columns]
        cols += [("spy_cierre", "SPY", "num")]
        cuerpo = f"<p>{len(r)} ruedas registradas desde {r.index.min():%d/%m/%Y}.</p>" + tabla(r.sort_index(ascending=False).head(30), cols)
    o = res.registro_opciones
    if not o.empty:
        cols = [("__indice__", "Fecha", "fecha")]
        for t in ("SPY", "QQQ", "IWM"):
            cols += [(f"{t}_pc_volumen", f"{t} put/call vol.", "num"), (f"{t}_pc_oi", f"{t} put/call OI", "num"),
                     (f"{t}_iv_atm_30", f"{t} IV ATM 30d", "pct0"), (f"{t}_skew_95_30", f"{t} skew 95 %", "pct"),
                     (f"{t}_gex_musd_1pct", f"{t} GEX (MUSD/1 %)", "int")]
        cols = [c for c in cols if c[0] == "__indice__" or c[0] in o.columns]
        cuerpo += "<h3>Snapshots de opciones (historia propia)</h3>" + tabla(o.sort_index(ascending=False).head(30), cols)
    return f"""
<p>Cada mañana hábil, antes de la apertura, se guarda una fila de la última rueda cerrada con todos los indicadores y puntajes tal como se
veían ese día (rama <code>registro</code> del repositorio). Nunca se reescriben filas pasadas: es el forward
test del sensor. También se guardan snapshots de cadenas de opciones y de ETFs para construir historia
propia de put/call, skew, GEX y flujos, que no existe gratis.</p>
{cuerpo}
"""


def seccion_datos(res: Resultados) -> str:
    filas = []
    for fuente, info in res.metadatos.items():
        filas.append({"fuente": fuente, "actualizado": info.get("actualizado_utc", ""),
                      "fallidos": ", ".join(info.get("fallidos", [])) or "—"})
    meta = tabla(pd.DataFrame(filas), [("fuente", "Fuente", "texto"), ("actualizado", "Actualizado (UTC)", "texto"),
                                       ("fallidos", "Fallidos", "texto")]) if filas else ""
    cal = res.calidad
    avisos = ""
    if not cal.empty and "estado" in cal.columns:
        malos = cal[cal["estado"] != "ok"]
        avisos = f"<p>{len(cal) - len(malos)} de {len(cal)} series sin avisos.</p>"
        if len(malos):
            avisos += tabla(malos, [("__indice__", "Ticker", "texto"), ("inicio", "Inicio", "texto"), ("fin", "Fin", "texto"),
                                    ("estado", "Aviso", "texto")])
    return f"""
<h3>Fuentes</h3>{meta}
<h3>Calidad de precios</h3>{avisos}
<h3>Descargas</h3>
<ul>
<li><a href="datos/mapa_SPY.csv">Mapa de comportamiento vs SPY (CSV)</a></li>
<li><a href="datos/episodios_SPY.csv">Tramos de caída de SPY con huella macro (CSV)</a></li>
<li><a href="datos/pilares.csv">Puntajes por pilar, historia completa (CSV)</a></li>
<li><a href="datos/historia_larga.csv">Industrias Fama-French en mercados bajistas (CSV)</a></li>
</ul>
<h3>Método y límites</h3>
<ul>
<li>Todo es gratuito: Yahoo Finance, FRED, CBOE y Kenneth French. Lo que no existe gratis se construye con
fórmulas propias o con snapshots diarios (ver catálogo de variables en el repositorio).</li>
<li>Los indicadores respetan la demora de publicación de cada serie y se normalizan solo con el pasado.</li>
<li>El puntaje de riesgo es preliminar: la validación predictiva (Fase 3) y el backtest del overlay (Fase 4) vienen después.</li>
<li>Pocos episodios grandes = poca potencia estadística. Por eso se muestran tablas por episodio y la historia desde 1926.</li>
<li>Nada de esto es una recomendación de inversión.</li>
</ul>
"""


# --- página ----------------------------------------------------------------------------------

PLANTILLA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compomercado</title>
<meta name="description" content="Sensor de comportamiento de mercado y sectores">
<script src="__PLOTLY__"></script>
<style>
:root {
  color-scheme: light;
  --plano: #f9f9f7; --superficie: #fcfcfb; --tinta: #0b0b0b; --tinta-2: #52514e; --muted: #898781;
  --grilla: #e1e0d9; --eje: #c3c2b7; --borde: rgba(11,11,11,0.10); --acento: #2a78d6; --barra: #b7d3f6;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plano: #0d0d0d; --superficie: #1a1a19; --tinta: #ffffff; --tinta-2: #c3c2b7; --muted: #898781;
    --grilla: #2c2c2a; --eje: #383835; --borde: rgba(255,255,255,0.10); --acento: #3987e5; --barra: #184f95;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plano: #0d0d0d; --superficie: #1a1a19; --tinta: #ffffff; --tinta-2: #c3c2b7; --muted: #898781;
  --grilla: #2c2c2a; --eje: #383835; --borde: rgba(255,255,255,0.10); --acento: #3987e5; --barra: #184f95;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--plano); color: var(--tinta); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 20px 16px 8px; max-width: 1280px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0; }
.sub { color: var(--tinta-2); margin: 4px 0 0; font-size: 13px; }
nav { position: sticky; top: 0; z-index: 10; background: var(--plano); border-bottom: 1px solid var(--borde); }
nav .inner { max-width: 1280px; margin: 0 auto; padding: 0 16px; display: flex; gap: 4px; overflow-x: auto; }
nav button, .subtab { background: none; border: 0; color: var(--tinta-2); padding: 10px 12px; font: inherit; cursor: pointer; white-space: nowrap; border-bottom: 2px solid transparent; }
nav button.activo, .subtab.activo { color: var(--tinta); border-bottom-color: var(--acento); font-weight: 600; }
main { max-width: 1280px; margin: 0 auto; padding: 8px 16px 48px; }
section { display: none; }
section.activo { display: block; }
.subpanel { display: none; } .subpanel.activo { display: block; }
h2 { font-size: 18px; margin: 20px 0 8px; }
h3 { font-size: 15px; margin: 24px 0 8px; }
h4 { font-size: 14px; margin: 16px 0 6px; color: var(--tinta-2); }
p { max-width: 900px; }
.pie { color: var(--tinta-2); font-size: 13px; margin: 4px 0 8px; }
.destacado { background: var(--superficie); border: 1px solid var(--borde); border-radius: 8px; padding: 10px 12px; }
.hero { display: flex; gap: 16px; flex-wrap: wrap; align-items: stretch; margin-top: 12px; }
.tile { background: var(--superficie); border: 1px solid var(--borde); border-radius: 12px; padding: 16px 20px; min-width: 240px; }
.tile-etq { color: var(--tinta-2); font-size: 13px; }
.tile-num { font-size: 48px; font-weight: 600; line-height: 1.1; }
.tile-de { font-size: 18px; color: var(--muted); font-weight: 400; }
.tile-sub { color: var(--tinta-2); font-size: 12px; margin-top: 6px; }
.estado { display: inline-flex; gap: 6px; align-items: center; font-weight: 600; margin-top: 6px; padding: 2px 10px; border-radius: 999px; border: 1px solid var(--borde); }
.estado::before { content: ""; width: 10px; height: 10px; border-radius: 50%; background: var(--c); }
.origen { font-size: 13px; color: var(--tinta-2); margin: 4px 0 0; }
.aviso-prueba { margin: 8px 0 0; padding: 8px 12px; border-radius: 8px; border: 2px solid var(--critical); color: var(--tinta); font-weight: 600; }
.estado.good { --c: var(--good); } .estado.warning { --c: var(--warning); } .estado.serious { --c: var(--serious); } .estado.critical { --c: var(--critical); }
.nota { flex: 1; min-width: 260px; background: var(--superficie); border: 1px solid var(--borde); border-radius: 12px; padding: 4px 16px; font-size: 14px; }
.dos { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.tabla-envoltura { overflow-x: auto; border: 1px solid var(--borde); border-radius: 8px; background: var(--superficie); max-height: 640px; overflow-y: auto; }
table.t { border-collapse: collapse; width: 100%; font-size: 13px; }
table.t th, table.t td { padding: 6px 10px; border-bottom: 1px solid var(--grilla); text-align: left; white-space: nowrap; }
table.t th { position: sticky; top: 0; background: var(--superficie); color: var(--tinta-2); font-weight: 600; cursor: pointer; user-select: none; }
table.t th[title] { text-decoration: underline dotted var(--muted); text-underline-offset: 3px; }
table.t .n { text-align: right; font-variant-numeric: tabular-nums; }
table.t tbody tr:hover { background: var(--plano); }
.barra { display: inline-block; width: 44px; height: 8px; margin-right: 6px; border-radius: 4px; vertical-align: middle;
  background: linear-gradient(90deg, var(--acento) var(--v), var(--grilla) var(--v)); }
.filtros { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; margin: 8px 0; }
.subtabs { display: flex; gap: 2px; overflow-x: auto; }
select { font: inherit; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--borde); background: var(--superficie); color: var(--tinta); }
code { font-size: 13px; }
a { color: var(--acento); }
</style>
</head>
<body>
<header>
  <h1>Compomercado</h1>
  <p class="sub">Sensor de comportamiento de mercado y sectores · datos al cierre del __FECHA__ · generado __GENERADO__ UTC</p>
  __ORIGEN__
</header>
<nav><div class="inner">__NAV__</div></nav>
<main>__SECCIONES__</main>
<script>
const TEMAS = {
  claro: {tinta: "#52514e", grilla: "#e1e0d9", eje: "#c3c2b7", medio: "#f0efec", borde: "#fcfcfb"},
  oscuro: {tinta: "#c3c2b7", grilla: "#2c2c2a", eje: "#383835", medio: "#383835", borde: "#1a1a19"},
};
function oscuro() {
  const t = document.documentElement.dataset.theme;
  if (t === "dark") return true;
  if (t === "light") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
function aplicarTema(div) {
  if (!div || !div.layout || !window.Plotly) return;
  const c = oscuro() ? TEMAS.oscuro : TEMAS.claro;
  const upd = {"font.color": c.tinta};
  for (const k of Object.keys(div.layout)) {
    if (/^[xy]axis\\d*$/.test(k)) { upd[k + ".gridcolor"] = c.grilla; upd[k + ".linecolor"] = c.eje; upd[k + ".tickcolor"] = c.eje; }
  }
  (div.layout.annotations || []).forEach((_, i) => { upd[`annotations[${i}].font.color`] = c.tinta; });
  Plotly.relayout(div, upd);
  div.data.forEach((tr, i) => {
    if (tr.meta === "divergente") {
      const cs = tr.colorscale.map(([p, col]) => [p, p === 0.5 ? c.medio : col]);
      Plotly.restyle(div, {colorscale: [cs]}, [i]);
    }
    if (tr.mode === "markers" && tr.marker && tr.marker.line) Plotly.restyle(div, {"marker.line.color": c.borde}, [i]);
  });
}
function temaTodos() { document.querySelectorAll(".plotly-graph-div").forEach(aplicarTema); }
function mostrar(id) {
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("activo", b.dataset.sec === id));
  document.querySelectorAll("main > section").forEach(s => s.classList.toggle("activo", s.id === id));
  document.querySelectorAll(`#${id} .plotly-graph-div`).forEach(d => { if (window.Plotly) Plotly.Plots.resize(d); });
  try { history.replaceState(null, "", "#" + id); } catch (e) {}
}
document.querySelectorAll("nav button").forEach(b => b.addEventListener("click", () => mostrar(b.dataset.sec)));
document.querySelectorAll(".subtab").forEach(b => b.addEventListener("click", () => {
  const cont = b.closest("section");
  cont.querySelectorAll(".subtab").forEach(x => x.classList.toggle("activo", x === b));
  cont.querySelectorAll(".subpanel").forEach(p => p.classList.toggle("activo", p.id === b.dataset.panel));
}));
document.querySelectorAll(".filtro-grupo").forEach(sel => sel.addEventListener("change", () => {
  sel.closest("section").querySelectorAll("table.filtrable tbody tr").forEach(tr => {
    tr.style.display = !sel.value || tr.dataset.grupo === sel.value ? "" : "none";
  });
}));
document.querySelectorAll("table.ordenable th").forEach((th, _) => th.addEventListener("click", () => {
  const tabla = th.closest("table"), i = [...th.parentNode.children].indexOf(th);
  const asc = th.dataset.orden !== "asc";
  tabla.querySelectorAll("th").forEach(x => delete x.dataset.orden);
  th.dataset.orden = asc ? "asc" : "desc";
  const filas = [...tabla.tBodies[0].rows];
  const val = tr => tr.cells[i].dataset.v;
  filas.sort((a, b) => {
    const x = val(a), y = val(b);
    if (x === "" && y === "") return 0; if (x === "") return 1; if (y === "") return -1;
    const nx = parseFloat(x), ny = parseFloat(y);
    const r = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y, "es");
    return asc ? r : -r;
  });
  filas.forEach(f => tabla.tBodies[0].appendChild(f));
}));
window.addEventListener("load", () => {
  temaTodos();
  const id = (location.hash || "").slice(1);
  if (id && document.getElementById(id)) mostrar(id);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", temaTodos);
</script>
</body>
</html>
"""

SECCIONES = [
    ("estado", "Estado del mercado", seccion_estado),
    ("ponderaciones", "Ponderaciones", seccion_ponderaciones),
    ("mapa", "Mapa de comportamiento", seccion_mapa),
    ("caidas", "Caídas", seccion_episodios),
    ("historia", "Historia desde 1926", seccion_historia),
    ("correlacion", "Correlaciones", seccion_correlacion),
    ("argentina", "Argentina", seccion_argentina),
    ("forward", "Registro forward", seccion_forward),
    ("datos", "Datos y método", seccion_datos),
]


def exportar_csv(res: Resultados, destino: Path) -> None:
    d = destino / "datos"
    d.mkdir(parents=True, exist_ok=True)
    if "SPY" in res.mapas:
        m = res.mapas["SPY"]
        t = m.tabla.copy()
        t.insert(0, "nombre", [res.nombres.get(i, i) for i in t.index])
        t.to_csv(d / "mapa_SPY.csv", float_format="%.6g")
        m.episodios.to_csv(d / "episodios_SPY.csv", float_format="%.6g")
    res.pilares.to_csv(d / "pilares.csv", float_format="%.2f")
    if res.historia_larga is not None:
        res.historia_larga.tabla.to_csv(d / "historia_larga.csv", float_format="%.6g")
    resumen = {"fecha": f"{res.fecha:%Y-%m-%d}",
               "total": None if res.pilares.empty else round(float(res.pilares["total"].dropna().iloc[-1]), 1)}
    (d / "resumen.json").write_text(json.dumps(resumen), encoding="utf-8")


def construir(res: Resultados, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    nav, secciones = [], []
    fallidas: list[str] = []
    for k, (id_, titulo, fn) in enumerate(SECCIONES):
        activo = " activo" if k == 0 else ""
        try:
            cuerpo = fn(res)
        except Exception as e:  # noqa: BLE001 - una sección rota no tira abajo el dashboard
            log.exception("Sección %s del dashboard falló", id_)
            fallidas.append(id_)
            cuerpo = f'<p class="destacado">Esta sección falló al construirse: {html.escape(str(e))}</p>'
        nav.append(f'<button class="{activo.strip()}" data-sec="{id_}">{titulo}</button>')
        secciones.append(f'<section id="{id_}" class="{activo.strip()}"><h2>{titulo}</h2>{cuerpo}</section>')
    o = res.origen or {}
    if o.get("real"):
        origen = (f'<p class="origen">✓ Datos reales descargados de Yahoo Finance, FRED, CBOE y Kenneth French '
                  f'({html.escape(str(o.get("descargado_utc") or "")[:16].replace("T", " "))} UTC).</p>')
    else:
        origen = ('<p class="aviso-prueba">✖ DATOS DE PRUEBA: este tablero no se armó con una descarga real de '
                  'las fuentes y no refleja el mercado.</p>')
    pagina = (PLANTILLA.replace("__PLOTLY__", PLOTLY_JS)
              .replace("__ORIGEN__", origen)
              .replace("__FECHA__", f"{res.fecha:%d/%m/%Y}")
              .replace("__GENERADO__", datetime.now(UTC).strftime("%Y-%m-%d %H:%M"))
              .replace("__NAV__", "".join(nav))
              .replace("__SECCIONES__", "".join(secciones)))
    ruta = destino / "index.html"
    ruta.write_text(pagina, encoding="utf-8")
    exportar_csv(res, destino)
    (destino / ".nojekyll").write_text("", encoding="utf-8")
    return ruta
