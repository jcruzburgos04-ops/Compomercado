"""Importancia de cada indicador y ponderación por ranking dentro de cada pilar y entre pilares.

Método (pensado para no engañarse con el backtest):

1. **Objetivos swing**: caída máxima de SPY de 3 % o más en 5 ruedas, de 5 % o más en 10 y de 5 %
   o más en 21 ruedas.
2. **Importancia** de un indicador = AUC de su percentil de riesgo contra esos objetivos (0,5 = no
   informa; 1 = separa perfecto). Se promedian los tres horizontes.
3. **Depuración**: queda afuera el indicador con poca historia, el que no informa (AUC < 0,52), el
   que funciona al revés de lo esperado (AUC < 0,48: no se le da vuelta el signo, porque un signo
   contrario a la teoría es sospechoso) y el redundante (correlación ≥ 0,85 con otro más importante
   del mismo pilar).
4. **Pesos por ranking**: dentro de cada pilar, el más importante pesa n, el segundo n−1, … el
   último 1 (normalizados). Los pilares pesan igual en el total: ponderarlos por ranking también se
   evalúa, pero con datos reales empeoró 2015–2019 fuera de muestra y no mejoró el resto.
5. **Walk-forward**: los pesos de cada año se calculan solo con datos hasta 21 ruedas antes del 1
   de enero (purga: el objetivo mira 21 ruedas hacia adelante). Así la serie ponderada es fuera de
   muestra y se puede comparar honestamente contra pesos iguales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .estado import PILARES, Indicador, caida_maxima_futura

log = logging.getLogger(__name__)

OBJETIVOS = {"y1": (5, 0.03), "y2": (10, 0.05), "y3": (21, 0.05)}
NOMBRES_OBJETIVOS = {"y1": "Caída ≥3 % en 5 ruedas", "y2": "Caída ≥5 % en 10 ruedas", "y3": "Caída ≥5 % en 21 ruedas"}
PURGA = 21
MIN_OBS = 750          # ruedas con indicador y objetivo para evaluar (≈3 años)
MIN_POSITIVOS = 20
AUC_INFORMA = 0.52
AUC_CONTRARIO = 0.48
CORR_REDUNDANTE = 0.85
MIN_PILARES = 3


# --- métricas -------------------------------------------------------------------------------

def objetivos(spy: pd.Series) -> pd.DataFrame:
    """1 si SPY cae al menos u en las próximas h ruedas, 0 si no, NaN si todavía no se sabe."""
    salida = {}
    for k, (h, u) in OBJETIVOS.items():
        fut = caida_maxima_futura(spy, h)
        salida[k] = (fut <= -u).astype(float).where(fut.notna())
    return pd.DataFrame(salida)


def _alinear(score: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([score, y], axis=1, keys=["s", "y"]).dropna()
    return df["s"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float).astype(bool)


def auc(score: pd.Series, y: pd.Series, min_pos: int = MIN_POSITIVOS) -> float:
    """Área bajo la curva ROC (Mann-Whitney): probabilidad de que un día con caída posterior tenga
    un puntaje más alto que un día sin caída."""
    s, t = _alinear(score, y)
    n1 = int(t.sum())
    n0 = len(t) - n1
    if n1 < min_pos or n0 < min_pos:
        return np.nan
    r = rankdata(s)
    return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precision_media(score: pd.Series, y: pd.Series) -> float:
    """Average precision (área bajo precisión-recall): la métrica relevante con eventos raros."""
    s, t = _alinear(score, y)
    n1 = int(t.sum())
    if n1 == 0:
        return np.nan
    orden = np.argsort(-s, kind="mergesort")
    tt = t[orden]
    precision = np.cumsum(tt) / np.arange(1, len(tt) + 1)
    return float((precision * tt).sum() / n1)


def precision_tope(score: pd.Series, y: pd.Series, cuantil: float = 0.8) -> tuple[float, float]:
    """Frecuencia del evento en el 20 % de días más riesgosos y en todos los días."""
    s, t = _alinear(score, y)
    if len(s) == 0:
        return np.nan, np.nan
    tope = s >= np.quantile(s, cuantil)
    return float(t[tope].mean()), float(t.mean())


# --- importancia y pesos --------------------------------------------------------------------

def importancia(riesgo: pd.DataFrame, Y: pd.DataFrame, hasta=None) -> pd.DataFrame:
    """AUC de cada columna de `riesgo` contra cada objetivo, con datos hasta `hasta` (inclusive)."""
    R = riesgo.loc[:hasta] if hasta is not None else riesgo
    Yh = Y.reindex(R.index)
    filas = {}
    for col in R.columns:
        fila = {f"auc_{k}": auc(R[col], Yh[k]) for k in OBJETIVOS}
        fila["n"] = int((R[col].notna() & Yh["y3"].notna()).sum())
        aucs = [fila[f"auc_{k}"] for k in OBJETIVOS if not np.isnan(fila[f"auc_{k}"])]
        fila["auc_media"] = float(np.mean(aucs)) if aucs else np.nan
        filas[col] = fila
    return pd.DataFrame.from_dict(filas, orient="index")


def pesos_por_ranking(importancias: pd.Series) -> pd.Series:
    """El más importante pesa n, el siguiente n−1, … el último 1; suman 1."""
    if importancias.empty:
        return pd.Series(dtype=float)
    orden = importancias.sort_values(ascending=False).index
    n = len(orden)
    w = pd.Series(np.arange(n, 0, -1, dtype=float), index=orden)
    return w / w.sum()


def _depurar(imp: pd.DataFrame, R: pd.DataFrame, ids: list[str]) -> tuple[list[str], dict[str, str]]:
    """Devuelve (incluidos por importancia descendente, {excluido: motivo})."""
    motivos: dict[str, str] = {}
    candidatos = []
    for i in ids:
        f = imp.loc[i] if i in imp.index else None
        if f is None or f["n"] < MIN_OBS or np.isnan(f["auc_media"]):
            motivos[i] = "poca historia"
        elif f["auc_media"] < AUC_CONTRARIO:
            motivos[i] = "funciona al revés de lo esperado"
        elif f["auc_media"] < AUC_INFORMA:
            motivos[i] = "no anticipa caídas"
        else:
            candidatos.append(i)
    candidatos.sort(key=lambda i: imp.loc[i, "auc_media"], reverse=True)
    incluidos: list[str] = []
    for i in candidatos:
        corr = R[incluidos + [i]].corr()[i].drop(i) if incluidos else pd.Series(dtype=float)
        parecido = corr[corr.abs() >= CORR_REDUNDANTE]
        if len(parecido):
            motivos[i] = f"repite a {parecido.abs().idxmax()}"
        else:
            incluidos.append(i)
    return incluidos, motivos


@dataclass
class Pesos:
    """Pesos vigentes y el detalle de por qué quedó cada indicador y cada pilar."""

    hasta: pd.Timestamp | None
    indicadores: pd.DataFrame  # índice id: pilar, auc_y1..y3, auc_media, n, peso, estado
    pilares: pd.DataFrame      # índice pilar: auc_media, peso, incluidos, total, estado
    peso_ind: dict[str, pd.Series] = field(default_factory=dict)  # pilar -> pesos (suman 1)
    peso_pilar: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def calcular_pesos(indicadores: list[Indicador], riesgo: pd.DataFrame, Y: pd.DataFrame, hasta=None) -> Pesos:
    compuestos = [i for i in indicadores if i.en_compuesto and i.id in riesgo.columns]
    R = riesgo.loc[:hasta] if hasta is not None else riesgo
    imp = importancia(R[[i.id for i in compuestos]], Y, None)

    filas_ind, peso_ind = {}, {}
    for pilar in PILARES:
        ids = [i.id for i in compuestos if i.pilar == pilar]
        if not ids:
            continue
        incluidos, motivos = _depurar(imp, R, ids)
        w = pesos_por_ranking(imp.loc[incluidos, "auc_media"]) if incluidos else pd.Series(dtype=float)
        if len(w):
            peso_ind[pilar] = w
        for i in ids:
            fila = imp.loc[i].to_dict() if i in imp.index else {}
            fila.update({"pilar": pilar, "peso": float(w.get(i, 0.0)), "estado": motivos.get(i, "incluido")})
            filas_ind[i] = fila
    tabla_ind = pd.DataFrame.from_dict(filas_ind, orient="index")

    # Importancia de cada pilar ya ponderado por dentro, y ranking entre pilares.
    pil = _pilares(R, peso_ind)
    imp_pil = importancia(pil, Y, None) if not pil.empty else pd.DataFrame()
    filas_pil = {}
    elegibles = []
    for pilar in PILARES:
        total = int((tabla_ind["pilar"] == pilar).sum()) if not tabla_ind.empty else 0
        incl = int(((tabla_ind["pilar"] == pilar) & (tabla_ind["peso"] > 0)).sum()) if total else 0
        f = imp_pil.loc[pilar] if pilar in imp_pil.index else None
        if f is None or incl == 0:
            estado = "sin indicadores que informen"
        elif f["n"] < MIN_OBS or np.isnan(f["auc_media"]):
            estado = "poca historia"
        elif f["auc_media"] < AUC_INFORMA:
            estado = "no anticipa caídas"
        else:
            estado = "incluido"
            elegibles.append(pilar)
        filas_pil[pilar] = {"auc_media": np.nan if f is None else f["auc_media"],
                            "incluidos": incl, "total": total, "estado": estado}
    peso_pil = pesos_por_ranking(imp_pil.loc[elegibles, "auc_media"]) if elegibles else pd.Series(dtype=float)
    tabla_pil = pd.DataFrame.from_dict(filas_pil, orient="index")
    tabla_pil["peso"] = [float(peso_pil.get(p, 0.0)) for p in tabla_pil.index]
    return Pesos(hasta=hasta, indicadores=tabla_ind, pilares=tabla_pil, peso_ind=peso_ind, peso_pilar=peso_pil)


# --- compuestos -----------------------------------------------------------------------------

def _promedio_ponderado(X: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Promedio ponderado fila por fila, renormalizando entre las columnas con dato ese día."""
    X = X.reindex(columns=w.index)
    W = X.notna().mul(w, axis=1)
    den = W.sum(axis=1)
    return (X.fillna(0).mul(w, axis=1).sum(axis=1) / den).where(den > 0)


def _pilares(riesgo: pd.DataFrame, peso_ind: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({p: _promedio_ponderado(riesgo, w) for p, w in peso_ind.items()})


def compuesto(riesgo: pd.DataFrame, pesos: Pesos, entre_pilares: bool = True) -> pd.DataFrame:
    """Pilares ponderados por dentro y total. Con `entre_pilares=False`, los pilares pesan igual."""
    pil = _pilares(riesgo, pesos.peso_ind)
    if pil.empty:
        return pil
    if entre_pilares and len(pesos.peso_pilar) >= 2:
        w = pesos.peso_pilar
        disponibles = pil[w.index].notna().sum(axis=1)
        pil["total"] = _promedio_ponderado(pil, w).where(disponibles >= min(MIN_PILARES, len(w)))
    else:
        pil["total"] = pil.mean(axis=1, skipna=True).where(pil.notna().sum(axis=1) >= MIN_PILARES)
    return pil


# --- walk-forward y evaluación ----------------------------------------------------------------

@dataclass
class Ponderacion:
    pilares: pd.DataFrame           # principal: pilares ponderados por dentro + total con pilares iguales
    total_entre: pd.Series          # alternativa evaluada: también ponderado entre pilares
    pesos_vigentes: Pesos           # los que se usan este año
    peso_pilar_total: pd.Series     # peso efectivo de cada pilar en el total (iguales)
    historial: pd.DataFrame         # año x pilar: importancia (AUC) estimada ese año
    evaluacion: pd.DataFrame        # variante x período: métricas fuera de muestra
    inicio_oos: pd.Timestamp | None


def walk_forward(indicadores: list[Indicador], riesgo: pd.DataFrame, Y: pd.DataFrame, primer_anio: int = 2005):
    fechas = riesgo.index
    partes, partes_entre, historial = [], [], {}
    vigentes = None
    for anio in range(primer_anio, fechas.max().year + 1):
        prueba = fechas[fechas.year == anio]
        if prueba.empty:
            continue
        pos = fechas.get_loc(prueba[0]) - PURGA - 1
        if pos < 0:
            continue
        pesos = calcular_pesos(indicadores, riesgo, Y, hasta=fechas[pos])
        if len(pesos.peso_pilar) < 2:
            continue
        # Se calcula sobre toda la historia hasta fin de año: los percentiles ya son point-in-time.
        dentro = compuesto(riesgo.loc[: prueba[-1]], pesos, entre_pilares=False)
        entre = compuesto(riesgo.loc[: prueba[-1]], pesos, entre_pilares=True)["total"]
        partes.append(dentro.loc[prueba])
        partes_entre.append(entre.loc[prueba])
        historial[anio] = pesos.pilares["auc_media"]
        vigentes = pesos
    if not partes:
        return pd.DataFrame(), pd.Series(dtype=float), None, pd.DataFrame()
    return (pd.concat(partes).sort_index(), pd.concat(partes_entre).sort_index(), vigentes,
            pd.DataFrame(historial).T.sort_index())


def evaluar(variantes: dict[str, pd.Series], Y: pd.DataFrame, periodos: dict[str, tuple]) -> pd.DataFrame:
    filas = []
    for nombre, s in variantes.items():
        for per, (desde, hasta) in periodos.items():
            ss = s.loc[desde:hasta].dropna()
            yy = Y.reindex(ss.index)
            if ss.empty or yy["y3"].notna().sum() < MIN_OBS // 3:
                continue
            tope, base = precision_tope(ss, yy["y3"])
            filas.append({
                "variante": nombre, "periodo": per, "ruedas": int(yy["y3"].notna().sum()),
                **{f"auc_{k}": auc(ss, yy[k]) for k in OBJETIVOS},
                "ap_y3": precision_media(ss, yy["y3"]),
                "tope20_y3": tope, "base_y3": base,
            })
    return pd.DataFrame(filas)


def analizar(indicadores: list[Indicador], riesgo: pd.DataFrame, total_igual: pd.Series,
             spy: pd.Series, baselines: dict[str, pd.Series], primer_anio: int = 2005) -> Ponderacion | None:
    Y = objetivos(spy.reindex(riesgo.index))
    pil, entre, vigentes, historial = walk_forward(indicadores, riesgo, Y, primer_anio)
    if vigentes is None or pil.empty:
        log.warning("Ponderación: no hay historia suficiente para estimar pesos")
        return None
    inicio = pil["total"].first_valid_index()
    fin = pil.index.max()
    periodos = {"Todo el período": (inicio, fin), "2005–2014": (inicio, "2014-12-31"),
                "2015–2019": ("2015-01-01", "2019-12-31"), "2020 en adelante": ("2020-01-01", fin)}
    variantes = {"Ponderado dentro de pilares (principal)": pil["total"],
                 "Ponderado dentro y entre pilares": entre, "Pesos iguales (anterior)": total_igual}
    variantes.update(baselines)
    ev = evaluar(variantes, Y, periodos)
    log.info("Ponderación: pesos vigentes estimados hasta %s", vigentes.hasta)
    activos = [p for p in pil.columns if p != "total"]
    peso_total = pd.Series(1 / len(activos), index=activos) if activos else pd.Series(dtype=float)
    return Ponderacion(pilares=pil, total_entre=entre, pesos_vigentes=vigentes, peso_pilar_total=peso_total,
                       historial=historial, evaluacion=ev, inicio_oos=inicio)
