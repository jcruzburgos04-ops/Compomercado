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


def mcclellan(precios: pd.DataFrame, minimo: int = 10) -> pd.Series:
    """Oscilador McClellan sobre un universo: EMA 19 − EMA 39 de (suben − bajan) / (suben + bajan)."""
    r = precios.pct_change(fill_method=None)
    suben, bajan = (r > 0).sum(axis=1), (r < 0).sum(axis=1)
    n = suben + bajan
    neto = ((suben - bajan) / n).where(n >= minimo)
    return neto.ewm(span=19, min_periods=19).mean() - neto.ewm(span=39, min_periods=39).mean()


def maximos_minimos(precios: pd.DataFrame, h: int = 252, suavizado: int = 10, minimo: int = 10) -> pd.Series:
    """% del universo en máximo de h ruedas menos % en mínimo, promedio de `suavizado` ruedas."""
    alto = precios.rolling(h, min_periods=h).max()
    bajo = precios.rolling(h, min_periods=h).min()
    validos = alto.notna() & precios.notna()
    n = validos.sum(axis=1)
    neto = ((precios >= alto) & validos).sum(axis=1) - ((precios <= bajo) & validos).sum(axis=1)
    return (neto / n).where(n >= minimo).rolling(suavizado, min_periods=1).mean()


def divergencia_amplitud(p: pd.Series, participacion: pd.Series, h: int = 21, cerca: float = 0.02,
                         umbral: float = 0.5) -> pd.Series:
    """Ruedas de las últimas `h` con el índice a menos de `cerca` de su máximo de 252 ruedas y menos de
    `umbral` del universo sobre su media de 50: el índice sube con pocos."""
    en_maximo = p >= p.rolling(252, min_periods=252).max() * (1 - cerca)
    debil = participacion < umbral
    return (en_maximo & debil).astype(float).where(participacion.notna()).rolling(h, min_periods=h).sum()


def choque(p: pd.Series, h: int = 21, ventana: int = 252) -> pd.Series:
    """Tamaño del movimiento de h ruedas medido en desvíos de su propia historia (valor absoluto)."""
    ret = np.log(p).diff(h)
    sd = np.log(p).diff().rolling(ventana, min_periods=ventana // 2).std() * np.sqrt(h)
    return (ret / sd).abs()


def ubicacion_cierre(cierre: pd.Series, maximo: pd.Series, minimo: pd.Series, h: int = 21) -> pd.Series:
    """Ubicación del cierre en el rango del día (CLV, −1 = en el mínimo, +1 = en el máximo), media h ruedas."""
    rango = (maximo - minimo).where(maximo > minimo)
    clv = ((cierre - minimo) - (maximo - cierre)) / rango
    return clv.rolling(h, min_periods=h // 2).mean()


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
    agregar("vix3m_vix6m", "VIX3M / VIX6M", "volatilidad", +1,
            lambda: S.vol("VIX3M").reindex(cal) / S.vol("VIX6M").reindex(cal), "x",
            "> 1 = el estrés se extiende a varios meses")
    agregar("skew", "CBOE SKEW", "volatilidad", +1, lambda: S.vol("SKEW").reindex(cal),
            desc="Precio relativo de la protección contra caídas extremas")
    agregar("vxn_vix", "VXN − VIX (tecnología)", "volatilidad", +1, lambda: S.vol("VXN").reindex(cal) - vix(),
            desc="Estrés concentrado en el Nasdaq 100")
    agregar("rvx_vix", "RVX − VIX (small caps)", "volatilidad", +1, lambda: S.vol("RVX").reindex(cal) - vix(),
            desc="Estrés concentrado en el Russell 2000")
    agregar("ovx", "OVX (vol. del petróleo)", "volatilidad", +1, lambda: S.vol("OVX").reindex(cal))
    agregar("gvz", "GVZ (vol. del oro)", "volatilidad", +1, lambda: S.vol("GVZ").reindex(cal))

    # Tendencia
    agregar("spy_vs_200", "SPY vs media de 200", "tendencia", -1, lambda: spy / spy.rolling(200).mean() - 1, "pct")
    agregar("drawdown", "Drawdown de SPY", "tendencia", -1, lambda: spy / spy.cummax() - 1, "pct")
    agregar("mom_63", "Momentum SPY 63d", "tendencia", -1, lambda: spy / spy.shift(63) - 1, "pct")
    agregar("spy_vs_50", "SPY vs media de 50", "tendencia", -1, lambda: spy / spy.rolling(50).mean() - 1, "pct")
    agregar("pendiente_200", "Pendiente de la media de 200 (21d)", "tendencia", -1,
            lambda: spy.rolling(200).mean().pct_change(21), "pct")
    agregar("mom_252", "Momentum SPY 12 meses", "tendencia", -1, lambda: spy / spy.shift(252) - 1, "pct")
    agregar("ret_ajustado", "Retorno 63d / volatilidad", "tendencia", -1,
            lambda: np.log(spy).diff(63) / (np.log(spy).diff().rolling(63).std() * np.sqrt(63)),
            desc="Calidad de la tendencia: suba con poca volatilidad = alto")
    ohlc = {k: S.alm.precios(k, ["SPY"]).reindex(cal).get("SPY") for k in ("apertura", "cierre", "maximo", "minimo")}
    if all(v is not None for v in ohlc.values()):
        ap, ci, mx, mn = ohlc["apertura"], ohlc["cierre"], ohlc["maximo"], ohlc["minimo"]
        agregar("overnight_intradia", "Intradía − overnight SPY (21d)", "tendencia", -1,
                lambda: (np.log(ci / ap).rolling(21).sum() - np.log(ap / ci.shift(1)).rolling(21).sum()), "pct",
                "Negativo = sube de noche y se vende en la sesión (posible distribución)")

    # Amplitud (universo propio de ETFs, sin sesgo de supervivencia)
    ind_sect = [t for t in SECTORES_BASE + [i.ticker for i in S.p.grupo("industrias")] if t in c.columns]
    glob = [i.ticker for i in S.p.grupo("global_etf") if i.ticker in c.columns]
    agregar("pct_200", "% sectores/industrias > media 200", "amplitud", -1,
            lambda: pct_sobre_media(c[ind_sect], 200), "pct")
    agregar("pct_50", "% sectores/industrias > media 50", "amplitud", -1,
            lambda: pct_sobre_media(c[ind_sect], 50), "pct")
    agregar("pct_20", "% sectores/industrias > media 20", "amplitud", -1,
            lambda: pct_sobre_media(c[ind_sect], 20), "pct")
    agregar("mcclellan", "McClellan (sectores/industrias)", "amplitud", -1, lambda: mcclellan(c[ind_sect]),
            desc="Momentum de la amplitud: EMA 19 − EMA 39 de suben menos bajan")
    agregar("max_min", "Máximos − mínimos de 52 semanas", "amplitud", -1, lambda: maximos_minimos(c[ind_sect]), "pct",
            "% del universo en máximos menos % en mínimos (media 10 ruedas)")
    agregar("divergencia", "Divergencia de amplitud (21 ruedas)", "amplitud", +1,
            lambda: divergencia_amplitud(spy, pct_sobre_media(c[ind_sect], 50)),
            desc="Ruedas con SPY a menos de 2 % del máximo y menos de la mitad del universo sobre su media de 50")
    agregar("rsp_spy", "Igual peso vs cap. (RSP/SPY 63d)", "amplitud", -1, lambda: rel(col("RSP"), spy, 63), "pct")
    agregar("mdy_spy", "Mid caps vs SPY (63d)", "amplitud", -1, lambda: rel(col("MDY"), spy, 63), "pct")
    agregar("xli_xlu", "Industriales vs utilities (63d)", "amplitud", -1, lambda: rel(col("XLI"), col("XLU"), 63), "pct")
    agregar("iwm_spy", "Small caps vs SPY (63d)", "amplitud", -1, lambda: rel(col("IWM"), spy, 63), "pct")
    agregar("xly_xlp", "Discrecional vs básico (63d)", "amplitud", -1, lambda: rel(col("XLY"), col("XLP"), 63), "pct")
    agregar("sphb_splv", "Alta beta vs baja vol. (63d)", "amplitud", -1, lambda: rel(col("SPHB"), col("SPLV"), 63), "pct")
    agregar("canarios", "Canarios vs SPY (63d)", "amplitud", -1,
            lambda: pd.concat([rel(col(t), spy, 63) for t in CANARIOS if t in c.columns], axis=1).mean(axis=1),
            "pct", "Promedio de semis, bancos regionales, transporte, retail y constructoras")

    # Amplitud sobre los componentes actuales del S&P 500: solo contexto (sesgo de supervivencia).
    sp = S.sp500()
    if sp.shape[1] >= 100:
        nota = "Componentes actuales del S&P 500: la historia tiene sesgo de supervivencia, no entra al puntaje"
        agregar("sp500_pct_200", "S&P 500: % sobre media 200", "amplitud", 0, lambda: pct_sobre_media(sp, 200, 100),
                "pct", nota)
        agregar("sp500_pct_50", "S&P 500: % sobre media 50", "amplitud", 0, lambda: pct_sobre_media(sp, 50, 100),
                "pct", nota)
        agregar("sp500_mcclellan", "S&P 500: McClellan", "amplitud", 0, lambda: mcclellan(sp, 100), desc=nota)
        agregar("sp500_max_min", "S&P 500: máximos − mínimos 52 sem.", "amplitud", 0,
                lambda: maximos_minimos(sp, minimo=100), "pct", nota)

    # Crédito
    agregar("hyg_ief", "High yield vs Tesoro (HYG/IEF 21d)", "credito", -1, lambda: rel(col("HYG"), col("IEF"), 21), "pct")
    agregar("baa10y", "Spread Baa − 10a", "credito", +1, lambda: S.fred("BAA10Y"))
    agregar("baa10y_d21", "Δ spread Baa 21d", "credito", +1, lambda: S.fred("BAA10Y").diff(21))
    agregar("stlfsi", "Estrés financiero St. Louis Fed", "credito", +1, lambda: S.fred("STLFSI4"))
    agregar("nfci", "Condiciones financieras (NFCI)", "credito", +1, lambda: S.fred("NFCI"))
    agregar("kre_spy", "Bancos regionales vs SPY (21d)", "credito", -1, lambda: rel(col("KRE"), spy, 21), "pct")
    agregar("lqd_ief", "Grado de inversión vs Tesoro (LQD/IEF 21d)", "credito", -1, lambda: rel(col("LQD"), col("IEF"), 21), "pct")
    agregar("emb_ief", "Deuda emergente vs Tesoro (EMB/IEF 21d)", "credito", -1, lambda: rel(col("EMB"), col("IEF"), 21), "pct")
    agregar("cp_spread", "Papel comercial − letras 3 meses", "credito", +1, lambda: S.fred("DCPF3M") - S.fred("DTB3"),
            desc="Costo de financiamiento de corto plazo de los bancos")
    agregar("liquidez_neta", "Liquidez neta de la Fed (63d)", "credito", -1,
            lambda: (S.fred("WALCL") / 1000 - S.fred("WTREGEN") - S.fred("RRPONTSYD").fillna(0)).pct_change(63), "pct",
            "Balance de la Fed − cuenta del Tesoro − repos reversos; variación en 63 ruedas")

    # Cross-asset y macro
    agregar("usdjpy_21", "USDJPY 21d (yen)", "cross_asset", -1, lambda: col("JPY=X") / col("JPY=X").shift(21) - 1,
            "pct", "Caída = yen fuerte: riesgo de desarme de carry")
    agregar("audjpy_21", "AUDJPY 21d", "cross_asset", -1, lambda: col("AUDJPY=X") / col("AUDJPY=X").shift(21) - 1, "pct")
    agregar("dxy_21", "Dólar (DXY) 21d", "cross_asset", +1, lambda: col("DX-Y.NYB") / col("DX-Y.NYB").shift(21) - 1, "pct")
    agregar("cobre_oro", "Cobre / oro (63d)", "cross_asset", -1, lambda: rel(col("HG=F"), col("GC=F"), 63), "pct")
    agregar("dgs10_d63", "Δ tasa 10a en 63d", "cross_asset", 0, lambda: S.fred("DGS10").diff(63),
            desc="Contexto: shock de tasas (+) o de crecimiento (−)")
    agregar("curva", "Curva 10a − 2a", "cross_asset", 0, lambda: S.fred("T10Y2Y"))
    agregar("curva_3m", "Curva 10a − 3m", "cross_asset", 0, lambda: S.fred("T10Y3M"))
    agregar("tasas_shock", "Shock de tasas 10a (21d, en σ)", "cross_asset", +1,
            lambda: S.fred("DGS10").diff(21) / (S.fred("DGS10").diff().rolling(252, min_periods=126).std() * np.sqrt(21)),
            desc="Suba de la tasa larga medida en desvíos de su historia")
    agregar("petroleo_shock", "Shock del petróleo (21d, en σ)", "cross_asset", +1, lambda: choque(col("CL=F")),
            desc="Movimiento grande del WTI en cualquier sentido")
    agregar("btc_21", "Bitcoin 21d", "cross_asset", -1, lambda: col("BTC-USD") / col("BTC-USD").shift(21) - 1, "pct",
            "Apetito de riesgo que cotiza todos los días")
    agregar("desempleo", "Pedidos de desempleo vs mínimo 52 sem.", "cross_asset", +1,
            lambda: S.fred("ICSA").rolling(20, min_periods=15).mean() / S.fred("ICSA").rolling(252, min_periods=200).min() - 1,
            "pct", "Promedio de 4 semanas contra su mínimo del último año")
    agregar("sahm", "Regla de Sahm", "cross_asset", +1, lambda: S.fred("SAHMREALTIME"),
            desc="≥ 0,5 = el desempleo subió como al inicio de una recesión")
    agregar("corr_acc_bonos", "Correlación SPY-TLT 63d", "cross_asset", +1,
            lambda: (spy.pct_change()).rolling(63).corr(col("TLT").pct_change()),
            desc="Positiva = los bonos no cubren")

    # Global
    agregar("pct_200_global", "% mercados globales > media 200", "global", -1, lambda: pct_sobre_media(c[glob], 200), "pct")
    agregar("eem_spy", "Emergentes vs SPY (63d)", "global", -1, lambda: rel(col("EEM"), spy, 63), "pct")
    agregar("efa_spy", "Desarrollados ex EE. UU. vs SPY (63d)", "global", 0, lambda: rel(col("EFA"), spy, 63), "pct",
            "Contexto: liderazgo global contra EE. UU.")
    agregar("china_21", "China vs SPY (FXI 21d)", "global", -1, lambda: rel(col("FXI"), spy, 21), "pct")
    agregar("usdcny_21", "USDCNY 21d (yuan)", "global", +1, lambda: col("CNY=X") / col("CNY=X").shift(21) - 1, "pct",
            "Suba = yuan débil: estrés en China")
    agregar("cew_21", "Monedas emergentes 21d (CEW)", "global", -1, lambda: col("CEW") / col("CEW").shift(21) - 1, "pct")
    europa = [t for t in ("^GDAXI", "^STOXX50E", "^FTSE", "^FCHI") if t in S.cierres.columns]
    if europa:
        agregar("europa_5d", "Europa 5 ruedas", "global", -1,
                lambda: pd.concat([S.precio(t).pct_change(5).reindex(cal, method="ffill") for t in europa], axis=1).mean(axis=1),
                "pct", "Promedio de DAX, Euro Stoxx 50, FTSE 100 y CAC 40")
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
    if all(v is not None for v in ohlc.values()):
        agregar("clv", "Ubicación del cierre SPY (21d)", "flujos", -1, lambda: ubicacion_cierre(ci, mx, mn),
                desc="−1 = cierra en el mínimo del día (venta en la sesión), +1 = en el máximo")
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
