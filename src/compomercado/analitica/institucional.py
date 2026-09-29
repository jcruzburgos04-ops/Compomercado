"""Qué muestran los flujos de bancos e instituciones, y si sirvieron para anticipar caídas.

Dos fuentes, las dos públicas y gratuitas:

- **CFTC, Traders in Financial Futures**: posición neta de dealers (bancos), asset managers
  (institucionales), fondos apalancados (hedge funds y CTAs) y operadores chicos en los futuros del
  S&P 500, Nasdaq, Russell, VIX, Tesoro, yen, euro, dólar y bitcoin. Semanal desde 2006.
- **Fed H.8 y H.4.1**: crédito, depósitos, préstamos a empresas y reservas de los bancos, ventanilla
  de descuento y la encuesta de condiciones de crédito.

Lectura de hoy: neto en % del interés abierto, cuántos desvíos se aparta de sus últimos 3 años (z) y
el cambio en 4 semanas. Capacidad de anticipar: la misma vara que el resto del sensor (AUC contra
caídas futuras de SPY, huella alrededor de los tramos de 5 %, y el total con y sin este pilar fuera de
muestra).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..datos.proveedores import cftc
from ..indicadores import ponderacion

GRUPOS = ["dealer", "am", "lev", "nr"]
PILAR = "institucional"


@dataclass
class Institucional:
    posiciones: pd.DataFrame                 # mercado x grupo: neto, z, índice COT, cambio 4 semanas
    fecha_informe: pd.Timestamp | None       # martes del último informe TFF
    fecha_conocido: pd.Timestamp | None      # desde cuándo se usa (martes + lag)
    netos: dict[str, pd.DataFrame]           # mercado -> semanas x grupo (neto % OI, fecha del informe)
    bancos: pd.DataFrame                     # indicadores de bancos: valor, percentil de riesgo, fecha
    importancia: pd.DataFrame                # indicador: AUC por objetivo, lectura
    extremos: list[str] = field(default_factory=list)   # frases para el parte


def posiciones(cot: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Posición de cada grupo en cada mercado en la última semana disponible (fecha del informe)."""
    filas, netos = [], {}
    for clave, m in cftc.MERCADOS.items():
        t = cftc.mercado(cot, clave)
        if t.empty or "oi" not in t:
            continue
        n = pd.DataFrame({g: cftc.neto(t, g) for g in GRUPOS if f"{g}_l" in t})
        netos[clave] = n
        for g in n.columns:
            s = n[g].dropna()
            if s.empty:
                continue
            z = cftc.z_movil(s)
            idx = cftc.indice_cot(s)
            filas.append({
                "mercado": m["nombre"], "clave": clave, "grupo": cftc.NOMBRES_CATEGORIAS[g], "g": g,
                "neto": float(s.iloc[-1]), "z": float(z.iloc[-1]) if z.notna().any() else np.nan,
                "indice": float(idx.iloc[-1]) if idx.notna().any() else np.nan,
                "cambio_4s": float(s.iloc[-1] - s.iloc[-5]) if len(s) > 4 else np.nan,
                "oi": float(t["oi"].iloc[-1]), "fecha": s.index[-1],
                "lectura": cftc.lectura(float(z.iloc[-1]) if z.notna().any() else np.nan),
            })
    return pd.DataFrame(filas), netos


def _frases(pos: pd.DataFrame) -> list[str]:
    """Posiciones extremas (|z| ≥ 2) para el parte."""
    if pos.empty:
        return []
    ext = pos[pos["z"].abs() >= 2].sort_values("z", key=lambda s: -s.abs())
    return [f"{r.grupo} en {r.mercado}: {r.lectura} (z {r.z:+.1f}, neto {r.neto:+.0%} del interés abierto)"
            for r in ext.itertuples()]


def analizar(cot_crudo: pd.DataFrame, lag_dias: int, indicadores: list, riesgo: pd.DataFrame,
             spy: pd.Series) -> Institucional | None:
    """`cot_crudo`: tabla TFF indexada por el martes del informe (sin lag). `indicadores`/`riesgo`:
    los del panel; se toman los del pilar institucional."""
    propios = [i for i in indicadores if i.pilar == PILAR]
    if cot_crudo.empty and not propios:
        return None
    pos, netos = posiciones(cot_crudo) if not cot_crudo.empty else (pd.DataFrame(), {})
    fecha = cot_crudo.index.max() if not cot_crudo.empty else None

    # Bancos: valor de hoy y percentil de riesgo.
    filas = []
    for ind in propios:
        if not ind.id.startswith("bancos_"):
            continue
        s = ind.serie.dropna()
        if s.empty:
            continue
        r = riesgo[ind.id].dropna() if ind.id in riesgo else pd.Series(dtype=float)
        filas.append({"id": ind.id, "nombre": ind.nombre, "valor": float(s.iloc[-1]), "formato": ind.formato,
                      "riesgo": float(r.iloc[-1]) if len(r) else np.nan, "fecha": s.index[-1],
                      "desc": ind.descripcion})
    bancos = pd.DataFrame(filas)

    # Capacidad de anticipar: AUC de cada indicador (orientado según su hipótesis) contra los objetivos.
    ids = [i.id for i in propios if i.id in riesgo.columns]
    imp = pd.DataFrame()
    if ids:
        Y = ponderacion.objetivos(spy.reindex(riesgo.index))
        imp = ponderacion.importancia(riesgo[ids], Y)
        imp["nombre"] = [next(i.nombre for i in propios if i.id == c) for c in imp.index]
        imp["desde"] = [riesgo[c].first_valid_index() for c in imp.index]

        def veredicto(a: float) -> str:
            if np.isnan(a):
                return "poca historia"
            if a >= 0.55:
                return "anticipa"
            if a >= ponderacion.AUC_INFORMA:
                return "anticipa poco"
            if a < 0.45:
                return "funciona al revés"
            if a < ponderacion.AUC_CONTRARIO:
                return "levemente al revés"
            return "no anticipa"

        imp["veredicto"] = [veredicto(a) for a in imp["auc_media"]]
        imp = imp.sort_values("auc_media", ascending=False)
    return Institucional(posiciones=pos, fecha_informe=fecha,
                         fecha_conocido=fecha + pd.Timedelta(days=lag_dias) if fecha is not None else None,
                         netos=netos, bancos=bancos, importancia=imp, extremos=_frases(pos))
