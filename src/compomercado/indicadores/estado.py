"""Panel de estado del mercado (primer corte de la Fase 2).

Cada indicador es una serie diaria en el calendario de EE. UU. que solo usa datos conocidos al
cierre de ese día. El puntaje por pilar es el promedio de los percentiles históricos
(expansivos, point-in-time) de sus indicadores, orientados para que 100 = más riesgo.

Es **descriptivo**: la calibración contra caídas futuras y la selección de variables se hacen en
la Fase 3. Por eso el puntaje no es una probabilidad.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..datos.series import Series

log = logging.getLogger(__name__)

MIN_HISTORIA = 504  # ruedas mínimas para calcular un percentil

PILARES = {
    "volatilidad": "Volatilidad y opciones",
    "tendencia": "Tendencia y momentum",
    "amplitud": "Amplitud e internals",
    "credito": "Crédito y financiamiento",
    "cross_asset": "Cross-asset y macro",
    "global": "Global",
    "correlacion": "Estructura de correlación",
    "flujos": "Flujos estimados",
}

SECTORES_BASE = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB"]
CANARIOS = ["SMH", "KRE", "IYT", "XRT", "ITB"]


@dataclass
class Indicador:
    id: str
    nombre: str
    pilar: str
    signo: int  # +1: más alto = más riesgo; -1: más bajo = más riesgo; 0: solo contexto
    serie: pd.Series
    formato: str = "num"  # num | pct | x
    descripcion: str = ""

    @property
    def en_compuesto(self) -> bool:
        return self.signo != 0


# --- utilidades ------------------------------------------------------------------------------

def percentil_expansivo(s: pd.Series, minimo: int = MIN_HISTORIA) -> pd.Series:
    """Percentil del valor de cada día dentro de toda su historia previa (incluido el día)."""
    s = s.dropna()
    return s.expanding(min_periods=minimo).rank(pct=True)


def rel(a: pd.Series, b: pd.Series, h: int) -> pd.Series:
    """Retorno relativo de a contra b en h ruedas."""
    return (a / a.shift(h)) / (b / b.shift(h)) - 1


def vol_realizada(p: pd.Series, h: int = 21) -> pd.Series:
    return np.log(p).diff().rolling(h).std() * np.sqrt(252)


def pct_sobre_media(precios: pd.DataFrame, h: int, minimo: int = 3) -> pd.Series:
    media = precios.rolling(h, min_periods=h).mean()
    validos = media.notna() & precios.notna()
    arriba = (precios > media) & validos
    n = validos.sum(axis=1)
    return (arriba.sum(axis=1) / n).where(n >= minimo)


def correlacion_promedio(r: pd.DataFrame, h: int = 63) -> pd.Series:
    """Correlación promedio entre pares, móvil."""
    salida = {}
    valores = r.to_numpy()
    for i in range(h, len(r) + 1):
        bloque = valores[i - h : i]
        bloque = bloque[:, ~np.isnan(bloque).any(axis=0)]
        if bloque.shape[1] < 3:
            continue
        c = np.corrcoef(bloque, rowvar=False)
        k = c.shape[0]
        salida[r.index[i - 1]] = (c.sum() - k) / (k * (k - 1))
    return pd.Series(salida)


def absorcion(r: pd.DataFrame, h: int = 252, k: int = 2) -> pd.Series:
    """Absorption ratio (Kritzman et al. 2011): varianza explicada por los k primeros componentes."""
    salida = {}
    valores = r.to_numpy()
    for i in range(h, len(r) + 1, 1):
        bloque = valores[i - h : i]
        bloque = bloque[:, ~np.isnan(bloque).any(axis=0)]
        if bloque.shape[1] <= k + 1:
            continue
        autovalores = np.linalg.eigvalsh(np.cov(bloque, rowvar=False))[::-1]
        salida[r.index[i - 1]] = autovalores[:k].sum() / autovalores.sum()
    return pd.Series(salida)


def turbulencia(r: pd.DataFrame, h: int = 252, suavizado: int = 10) -> pd.Series:
    """Turbulencia (Kritzman y Li 2010): distancia de Mahalanobis de los retornos del día contra la
    media y covarianza de las `h` ruedas previas. Se suaviza con una media de `suavizado` ruedas."""
    salida = {}
    valores = r.to_numpy()
    for i in range(h, len(r)):
        hist = valores[i - h : i]
        hoy = valores[i]
        cols = ~np.isnan(hist).any(axis=0) & ~np.isnan(hoy)
        if cols.sum() < 3:
            continue
        hist, hoy = hist[:, cols], hoy[cols]
        dif = hoy - hist.mean(axis=0)
        cov = np.cov(hist, rowvar=False)
        salida[r.index[i]] = float(dif @ np.linalg.pinv(cov) @ dif) / cols.sum()
    return pd.Series(salida).rolling(suavizado, min_periods=1).mean()


def dias_distribucion(cierre: pd.Series, volumen: pd.Series, h: int = 25, caida: float = -0.002) -> pd.Series:
    """Días de distribución (IBD): caída >= 0,2 % con más volumen que la rueda anterior, en h ruedas."""
    r = cierre / cierre.shift(1) - 1
    dist = (r <= caida) & (volumen > volumen.shift(1))
    return dist.astype(float).rolling(h).sum()


def cta_estimado(p: pd.Series, horizontes=(21, 63, 126, 252)) -> tuple[pd.Series, pd.Series]:
    """Posición estimada de CTAs (promedio de señales de tendencia, -1 a 1) y distancia al gatillo
    más cercano (variación % del precio que daría vuelta alguna señal)."""
    senales = pd.concat([np.sign(p / p.shift(h) - 1) for h in horizontes], axis=1)
    posicion = senales.mean(axis=1)
    distancias = pd.concat([p.shift(h) / p - 1 for h in horizontes], axis=1)
    gatillo = distancias.abs().min(axis=1) * np.sign(-posicion.replace(0, 1))
    return posicion, gatillo


def exposicion_vol_control(p: pd.Series, objetivo: float = 0.10, tope: float = 1.5) -> pd.Series:
    """Exposición estimada de fondos de control de volatilidad: min(tope, objetivo / vol realizada)."""
    vol = pd.concat([vol_realizada(p, 21), vol_realizada(p, 63)], axis=1).max(axis=1)
    return (objetivo / vol).clip(upper=tope)


# --- construcción de indicadores ------------------------------------------------------------

def calcular(S: Series) -> list[Indicador]:
    cal = S.calendario
    c = S.en_calendario()
    col = lambda t: c[t] if t in c.columns else pd.Series(np.nan, index=cal)  # noqa: E731
    spy = col("SPY")
    salida: list[Indicador] = []

    def agregar(id_, nombre, pilar, signo, fn: Callable[[], pd.Series], formato="num", desc=""):
        try:
            s = fn()
            s = s.reindex(cal).astype(float)
            if s.dropna().empty:
                log.info("Indicador %s sin datos", id_)
                return
            salida.append(Indicador(id_, nombre, pilar, signo, s, formato, desc))
        except Exception as e:  # noqa: BLE001 - un indicador roto no frena el resto
            log.warning("Indicador %s falló: %s", id_, e)

    vix = lambda: S.vol("VIX").reindex(cal)  # noqa: E731
    rv21 = vol_realizada(spy, 21)

    # Volatilidad
    agregar("vix", "VIX", "volatilidad", +1, vix, desc="Volatilidad implícita del S&P 500 a 30 días")
    agregar("vix_vix3m", "VIX / VIX3M", "volatilidad", +1,
            lambda: vix() / S.vol("VIX3M").reindex(cal), "x", "> 1 = estructura invertida (estrés agudo)")
    agregar("vix9d_vix", "VIX9D / VIX", "volatilidad", +1,
            lambda: S.vol("VIX9D").reindex(cal) / vix(), "x", "> 1 = pánico de muy corto plazo")
    agregar("vvix", "VVIX", "volatilidad", +1, lambda: S.vol("VVIX").reindex(cal), desc="Volatilidad del VIX")
    agregar("rv21", "Vol. realizada SPY 21d", "volatilidad", +1, lambda: rv21, "pct")
    agregar("vrp", "Prima de vol. (VIX − realizada)", "volatilidad", -1, lambda: vix() - rv21 * 100,
            desc="Negativa = la volatilidad real supera a la implícita (estrés en curso)")
    agregar("move", "MOVE (vol. de bonos)", "volatilidad", +1, lambda: col("^MOVE"))

    # Tendencia
    agregar("spy_vs_200", "SPY vs media de 200", "tendencia", -1, lambda: spy / spy.rolling(200).mean() - 1, "pct")
    agregar("drawdown", "Drawdown de SPY", "tendencia", -1, lambda: spy / spy.cummax() - 1, "pct")
    agregar("mom_63", "Momentum SPY 63d", "tendencia", -1, lambda: spy / spy.shift(63) - 1, "pct")

    # Amplitud (universo propio de ETFs, sin sesgo de supervivencia)
    ind_sect = [t for t in SECTORES_BASE + [i.ticker for i in S.p.grupo("industrias")] if t in c.columns]
    glob = [i.ticker for i in S.p.grupo("global_etf") if i.ticker in c.columns]
    agregar("pct_200", "% sectores/industrias > media 200", "amplitud", -1,
            lambda: pct_sobre_media(c[ind_sect], 200), "pct")
    agregar("pct_50", "% sectores/industrias > media 50", "amplitud", -1,
            lambda: pct_sobre_media(c[ind_sect], 50), "pct")
    agregar("rsp_spy", "Igual peso vs cap. (RSP/SPY 63d)", "amplitud", -1, lambda: rel(col("RSP"), spy, 63), "pct")
    agregar("iwm_spy", "Small caps vs SPY (63d)", "amplitud", -1, lambda: rel(col("IWM"), spy, 63), "pct")
    agregar("xly_xlp", "Discrecional vs básico (63d)", "amplitud", -1, lambda: rel(col("XLY"), col("XLP"), 63), "pct")
    agregar("sphb_splv", "Alta beta vs baja vol. (63d)", "amplitud", -1, lambda: rel(col("SPHB"), col("SPLV"), 63), "pct")
    agregar("canarios", "Canarios vs SPY (63d)", "amplitud", -1,
            lambda: pd.concat([rel(col(t), spy, 63) for t in CANARIOS if t in c.columns], axis=1).mean(axis=1),
            "pct", "Promedio de semis, bancos regionales, transporte, retail y constructoras")

    # Crédito
    agregar("hyg_ief", "High yield vs Tesoro (HYG/IEF 21d)", "credito", -1, lambda: rel(col("HYG"), col("IEF"), 21), "pct")
    agregar("baa10y", "Spread Baa − 10a", "credito", +1, lambda: S.fred("BAA10Y"))
    agregar("baa10y_d21", "Δ spread Baa 21d", "credito", +1, lambda: S.fred("BAA10Y").diff(21))
    agregar("stlfsi", "Estrés financiero St. Louis Fed", "credito", +1, lambda: S.fred("STLFSI4"))
    agregar("nfci", "Condiciones financieras (NFCI)", "credito", +1, lambda: S.fred("NFCI"))
    agregar("kre_spy", "Bancos regionales vs SPY (21d)", "credito", -1, lambda: rel(col("KRE"), spy, 21), "pct")

    # Cross-asset y macro
    agregar("usdjpy_21", "USDJPY 21d (yen)", "cross_asset", -1, lambda: col("JPY=X") / col("JPY=X").shift(21) - 1,
            "pct", "Caída = yen fuerte: riesgo de desarme de carry")
    agregar("audjpy_21", "AUDJPY 21d", "cross_asset", -1, lambda: col("AUDJPY=X") / col("AUDJPY=X").shift(21) - 1, "pct")
    agregar("dxy_21", "Dólar (DXY) 21d", "cross_asset", +1, lambda: col("DX-Y.NYB") / col("DX-Y.NYB").shift(21) - 1, "pct")
    agregar("cobre_oro", "Cobre / oro (63d)", "cross_asset", -1, lambda: rel(col("HG=F"), col("GC=F"), 63), "pct")
    agregar("dgs10_d63", "Δ tasa 10a en 63d", "cross_asset", 0, lambda: S.fred("DGS10").diff(63),
            desc="Contexto: shock de tasas (+) o de crecimiento (−)")
    agregar("curva", "Curva 10a − 2a", "cross_asset", 0, lambda: S.fred("T10Y2Y"))
    agregar("corr_acc_bonos", "Correlación SPY-TLT 63d", "cross_asset", +1,
            lambda: (spy.pct_change()).rolling(63).corr(col("TLT").pct_change()),
            desc="Positiva = los bonos no cubren")

    # Global
    agregar("pct_200_global", "% mercados globales > media 200", "global", -1, lambda: pct_sobre_media(c[glob], 200), "pct")
    agregar("eem_spy", "Emergentes vs SPY (63d)", "global", -1, lambda: rel(col("EEM"), spy, 63), "pct")
    asia = [t for t in ("^N225", "^HSI", "^KS11", "^TWII") if t in S.cierres.columns]
    agregar("asia_5d", "Asia 5 ruedas", "global", -1,
            lambda: pd.concat([S.precio(t).pct_change(5).reindex(cal, method="ffill") for t in asia], axis=1).mean(axis=1),
            "pct", "Promedio de Nikkei, Hang Seng, Kospi y Taiwán")

    # Estructura de correlación (9 sectores con historia desde 1998)
    base = [t for t in SECTORES_BASE if t in c.columns]
    r_sect = c[base].pct_change()
    agregar("corr_prom", "Correlación promedio entre sectores 63d", "correlacion", +1, lambda: correlacion_promedio(r_sect, 63))
    agregar("absorcion", "Absorption ratio (2 de 9 componentes)", "correlacion", +1, lambda: absorcion(r_sect, 252, 2), "pct")
    agregar("turbulencia", "Turbulencia (Mahalanobis, 10d)", "correlacion", +1, lambda: turbulencia(r_sect, 252, 10))

    # Flujos estimados desde precios y volumen
    vol_spy = S.alm.precios("volumen", ["SPY"]).reindex(cal)
    if "SPY" in vol_spy.columns:
        agregar("distribucion", "Días de distribución (25 ruedas)", "flujos", +1,
                lambda: dias_distribucion(spy, vol_spy["SPY"]))
    posicion, gatillo = cta_estimado(spy)
    agregar("cta", "Posición estimada de CTAs", "flujos", -1, lambda: posicion,
            desc="-1 = todos los plazos vendidos, +1 = todos comprados")
    agregar("cta_gatillo", "Distancia al gatillo CTA", "flujos", 0, lambda: gatillo, "pct",
            desc="Variación de SPY que daría vuelta la señal más cercana")
    agregar("vol_control", "Exposición fondos vol-control", "flujos", -1, lambda: exposicion_vol_control(spy), "x",
            desc="Estimada con objetivo 10 % y tope 1,5x")
    return salida


def puntajes(indicadores: list[Indicador]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (percentiles de riesgo por indicador, puntajes por pilar + total), en 0-100."""
    riesgo = {}
    for ind in indicadores:
        if not ind.en_compuesto:
            continue
        pct = percentil_expansivo(ind.serie)
        riesgo[ind.id] = (pct if ind.signo > 0 else 1 - pct) * 100
    riesgo_df = pd.DataFrame(riesgo)
    pilares = {}
    for pilar in PILARES:
        ids = [i.id for i in indicadores if i.pilar == pilar and i.id in riesgo_df.columns]
        if ids:
            pilares[pilar] = riesgo_df[ids].mean(axis=1, skipna=True)
    pil = pd.DataFrame(pilares)
    pil["total"] = pil.mean(axis=1, skipna=True).where(pil.notna().sum(axis=1) >= 3)
    return riesgo_df, pil


def caida_maxima_futura(p: pd.Series, h: int) -> pd.Series:
    """min(p[t+1..t+h]) / p[t] - 1 (objetivo; usa el futuro, solo para evaluar)."""
    futuro_min = p.shift(-1)[::-1].rolling(h, min_periods=h).min()[::-1]
    return futuro_min / p - 1


def tabla_base(total: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """Qué pasó después según el nivel del puntaje total (descriptivo, dentro de la muestra)."""
    df = pd.DataFrame(
        {
            "total": total,
            "y1": caida_maxima_futura(spy, 5) <= -0.03,
            "y2": caida_maxima_futura(spy, 10) <= -0.05,
            "y3": caida_maxima_futura(spy, 21) <= -0.05,
            "ret21": spy.shift(-21) / spy - 1,
        }
    ).dropna()
    df["tramo"] = pd.cut(df["total"], [0, 20, 40, 60, 80, 100], include_lowest=True,
                         labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    t = df.groupby("tramo", observed=False).agg(
        ruedas=("total", "size"),
        caida3_5d=("y1", "mean"),
        caida5_10d=("y2", "mean"),
        caida5_21d=("y3", "mean"),
        ret21_medio=("ret21", "mean"),
    )
    base = pd.DataFrame(
        {"ruedas": [len(df)], "caida3_5d": [df["y1"].mean()], "caida5_10d": [df["y2"].mean()],
         "caida5_21d": [df["y3"].mean()], "ret21_medio": [df["ret21"].mean()]},
        index=["Todas"],
    )
    return pd.concat([t, base])
