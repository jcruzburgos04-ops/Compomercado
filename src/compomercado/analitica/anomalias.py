"""Anomalías de mercado: detectarlas el mismo día y medir si sirvieron para algo.

Dos familias:

1. **Anomalías de comportamiento** (días raros): un movimiento, un volumen, una combinación de
   activos o una posición que se sale de su propia historia. Cada detector marca un día solo con
   datos conocidos al cierre de ese día (umbrales contra la historia previa, percentiles expansivos).
   Se evalúa como un evento:
   - **Probabilidad de caída** ≥ 5 % en las 21 ruedas siguientes después del evento contra la de
     cualquier día (lift), con test binomial y corrección de Benjamini-Hochberg porque se prueban
     muchos detectores. Los eventos se cuentan por episodio (un día marcado después de 21 ruedas sin
     marcas) para que las ventanas no se pisen.
   - **Desde máximos**: la misma cuenta solo con eventos que aparecen con SPY a menos de 3 % de su
     máximo de 63 ruedas. Separa lo que anticipa desde la calma de lo que acompaña una caída en curso.
   - **Cobertura**: qué parte de los tramos bajistas de 5 % tuvo el detector encendido el mes previo
     al pico (contra lo que daría el azar) y cuánto se concentra dentro de las caídas.
   - Retornos posteriores (21 y 63 ruedas) contra los de cualquier día: una anomalía puede marcar
     estrés y a la vez un buen punto de compra.
2. **Anomalías de calendario y factores** (las clásicas de la literatura): cambio de mes, lunes,
   pre-feriado, Halloween, rally de fin de año, septiembre, tamaño, valor y momentum. Se miden desde
   1926 por era para ver si siguen vigentes o desaparecieron después de publicarse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_ind

from ..datos.series import Series
from ..indicadores import ponderacion
from ..indicadores.estado import SECTORES_BASE, mcclellan, percentil_expansivo, rel, turbulencia
from .huellas import benjamini_hochberg, posiciones

log = logging.getLogger(__name__)

REFRACTARIO = 21          # ruedas sin marcas para que empiece un episodio nuevo
VENTANA_ANTES = 21        # "antes del pico" = las 21 ruedas previas (incluido el pico)
CERCA_MAXIMO = 0.97       # "desde máximos": SPY a menos de 3 % de su máximo de 63 ruedas
MIN_EPISODIOS = 8
Q_SIGNIFICATIVO = 0.10
LIFT_MINIMO = 1.25
RUEDAS_ACTIVA = 5         # activa hoy = marcó en las últimas 5 ruedas
VENTANA_CONTEO = 10
ACTIVOS_GLOBALES = ["SPY", "EFA", "EEM", "TLT", "GLD", "UUP", "HYG", "DBC"]

ERAS = {"1926–1962": ("1926-01-01", "1962-12-31"), "1963–1992": ("1963-01-01", "1992-12-31"),
        "1993–2009": ("1993-01-01", "2009-12-31"), "2010–hoy": ("2010-01-01", None)}


@dataclass
class Detector:
    id: str
    nombre: str
    familia: str
    descripcion: str
    serie: pd.Series          # 1 = anomalía ese día, 0 = no, NaN = no se puede calcular


# --- detectores ------------------------------------------------------------------------------------

def _marca(cond: pd.Series, valido: pd.Series) -> pd.Series:
    return cond.astype(float).where(valido.astype(bool))


def _sigma_previa(r: pd.Series, h: int = 252) -> pd.Series:
    """Desvío de las h ruedas anteriores (sin incluir el día)."""
    return r.rolling(h, min_periods=h // 2).std().shift(1)


def _primer_dia(cond: pd.Series, valido: pd.Series, h: int = 21) -> pd.Series:
    """1 el primer día en que se cumple después de h ruedas sin cumplirse."""
    c = cond.fillna(False).astype(bool)
    previo = c.shift(1, fill_value=False).astype(float).rolling(h, min_periods=1).max().astype(bool)
    return _marca(c & ~previo, valido)


def detectar(S: Series, precios: pd.DataFrame, indicadores: list | None = None) -> list[Detector]:
    """Todos los detectores sobre el calendario de EE. UU. Un detector sin datos no aparece."""
    cal = S.calendario
    col = lambda t: precios[t] if t in precios.columns else pd.Series(np.nan, index=cal)  # noqa: E731
    spy = col("SPY")
    r = np.log(spy).diff()
    sd = _sigma_previa(r)
    ret = spy.pct_change(fill_method=None)
    vix = S.vol("VIX").reindex(cal)
    salida: list[Detector] = []

    def agregar(id_, nombre, familia, desc, fn):
        try:
            s = fn().reindex(cal)
            if s.dropna().empty or s.sum() == 0:
                log.info("Anomalía %s sin datos o sin eventos", id_)
                return
            salida.append(Detector(id_, nombre, familia, desc, s))
        except Exception as e:  # noqa: BLE001 - un detector roto no frena el resto
            log.warning("Anomalía %s falló: %s", id_, e)

    # Precio y volumen de SPY
    agregar("panico", "Día de pánico (−3σ)", "precio",
            "SPY cae más de 3 desvíos de su último año en una rueda", lambda: _marca(r <= -3 * sd, sd.notna()))
    vol = S.alm.precios("volumen", ["SPY"]).reindex(cal).get("SPY", pd.Series(np.nan, index=cal))
    v50 = vol.rolling(50, min_periods=40).mean().shift(1)
    agregar("venta_volumen", "Venta con volumen extremo", "precio",
            "SPY cae 1 % o más con el doble del volumen medio de 50 ruedas",
            lambda: _marca((ret <= -0.01) & (vol >= 2 * v50), v50.notna() & (vol > 0)))
    ohlc = S.alm.ohlc_ajustado("SPY").reindex(cal) if "SPY" in S.cierres.columns else pd.DataFrame()
    if {"apertura", "maximo", "cierre"} <= set(ohlc.columns):
        gap = ohlc["apertura"] / ohlc["cierre"].shift(1) - 1
        sdg = _sigma_previa(gap)
        agregar("gap_bajista", "Gap bajista extremo (−3σ)", "precio",
                "SPY abre más de 3 desvíos por debajo del cierre anterior",
                lambda: _marca(gap <= -3 * sdg, sdg > 0.001))
        maximo_previo = ohlc["maximo"].shift(1).rolling(252, min_periods=252).max()
        agregar("reversion_clave", "Reversión en máximos (key reversal)", "precio",
                "SPY marca máximo de 52 semanas en el día pero cierra por debajo de la rueda anterior",
                lambda: _marca((ohlc["maximo"] >= maximo_previo) & (ohlc["cierre"] < ohlc["cierre"].shift(1)),
                               maximo_previo.notna()))

    # Volatilidad
    dvix = vix / vix.shift(1) - 1
    agregar("vix_y_spy_suben", "VIX y SPY suben juntos", "volatilidad",
            "SPY sube 0,2 % o más y el VIX sube 5 % o más: se compra cobertura en plena suba",
            lambda: _marca((ret >= 0.002) & (dvix >= 0.05), dvix.notna() & ret.notna()))
    agregar("vix_invertido", "Curva del VIX se invierte", "volatilidad",
            "VIX supera al VIX3M por primera vez en 21 ruedas",
            lambda: _primer_dia(vix / S.vol("VIX3M").reindex(cal) > 1, S.vol("VIX3M").reindex(cal).notna()))
    agregar("vix_minimo", "VIX en mínimo de un año", "volatilidad",
            "El VIX cierra en su mínimo de 252 ruedas (complacencia)",
            lambda: _marca(vix <= vix.shift(1).rolling(252, min_periods=252).min(),
                           vix.shift(1).rolling(252, min_periods=252).min().notna()))

    # Crédito
    hy = rel(col("HYG"), col("IEF"), 21)
    pct_hy = percentil_expansivo(hy).reindex(cal)
    cerca_max = spy >= 0.98 * spy.rolling(252, min_periods=252).max()
    agregar("credito_no_acompana", "El crédito no acompaña al máximo", "crédito",
            "SPY a menos de 2 % de su máximo de 52 semanas con el high yield en el 10 % más débil de su historia",
            lambda: _marca(cerca_max & (pct_hy <= 0.10), pct_hy.notna()))
    ret_tlt = col("TLT").pct_change(fill_method=None)
    agregar("bonos_y_acciones_caen", "Caen acciones y bonos juntos", "crédito",
            "SPY y TLT caen 1 % o más en la misma rueda: no hay refugio (liquidez o inflación)",
            lambda: _marca((ret <= -0.01) & (ret_tlt <= -0.01), ret_tlt.notna() & ret.notna()))

    # Estructura y global
    base = [t for t in SECTORES_BASE if t in precios.columns]
    if len(base) >= 5:
        t_dia = turbulencia(precios[base].pct_change(fill_method=None), 252, suavizado=1).reindex(cal)
        p_t = percentil_expansivo(t_dia).reindex(cal)
        agregar("turbulencia_extrema", "Turbulencia extrema entre sectores", "estructura",
                "Los retornos de los sectores se alejan de su patrón habitual (Mahalanobis en el 1 % más alto)",
                lambda: _marca(p_t >= 0.99, p_t.notna()))
    glob = [t for t in ACTIVOS_GLOBALES if t in precios.columns]
    if len(glob) >= 5:
        t_g = turbulencia(precios[glob].pct_change(fill_method=None), 252, suavizado=1).reindex(cal)
        p_g = percentil_expansivo(t_g).reindex(cal)
        agregar("anomalia_global", "Anomalía cross-asset", "estructura",
                "Acciones, bonos, oro, dólar, crédito y commodities se mueven de forma inusual juntos "
                "(Mahalanobis en el 1 % más alto)", lambda: _marca(p_g >= 0.99, p_g.notna()))
    jpy = np.log(col("JPY=X"))
    sd_j = _sigma_previa(jpy.diff()) * np.sqrt(5)
    agregar("yen_shock", "Shock del yen", "estructura",
            "El yen se aprecia más de 2,5 desvíos en 5 ruedas (posible desarme de carry trade)",
            lambda: _marca(jpy.diff(5) <= -2.5 * sd_j, sd_j.notna()))

    # Amplitud (universo de ETFs propio, sin sesgo de supervivencia)
    univ = [t for t in SECTORES_BASE + [i.ticker for i in S.p.grupo("industrias")] if t in precios.columns]
    if len(univ) >= 10:
        r_u = precios[univ].pct_change(fill_method=None)
        n_u = r_u.notna().sum(axis=1)
        suben = (r_u > 0).sum(axis=1) / n_u.where(n_u > 0)
        agregar("suba_sin_amplitud", "Suba sin amplitud", "amplitud",
                "SPY sube 0,5 % o más pero sube menos del 40 % de sectores e industrias",
                lambda: _marca((ret >= 0.005) & (suben < 0.40), n_u >= 10))
    ancho = [t for t in univ + [i.ticker for i in S.p.grupo("factores")] + [i.ticker for i in S.p.grupo("global_etf")]
             if t in precios.columns]
    if len(ancho) >= 20:
        pa = precios[ancho]
        alto = pa >= pa.rolling(252, min_periods=252).max()
        bajo = pa <= pa.rolling(252, min_periods=252).min()
        validos = pa.rolling(252, min_periods=252).max().notna()
        n_a = validos.sum(axis=1)
        s_alto = (alto & validos).sum(axis=1) / n_a.where(n_a > 0)
        s_bajo = (bajo & validos).sum(axis=1) / n_a.where(n_a > 0)
        mc = mcclellan(pa, 10)
        agregar("hindenburg", "Hindenburg (adaptado a ETFs)", "amplitud",
                "Muchos ETFs en máximos y en mínimos de 52 semanas el mismo día (≥ 2,8 % cada uno), SPY sobre su "
                "media de 50 y McClellan negativo: mercado dividido",
                lambda: _marca((s_alto >= 0.028) & (s_bajo >= 0.028) & (spy > spy.rolling(50).mean()) & (mc < 0),
                               (n_a >= 20) & mc.notna()))

    # Institucional (CFTC y bancos): posiciones extremas o bancos pidiendo liquidez
    por_id = {i.id: i.serie for i in (indicadores or [])}
    if "cot_lev_vix" in por_id:
        z = por_id["cot_lev_vix"].reindex(cal)
        agregar("cot_vix_extremo", "Fondos muy vendidos de VIX", "institucional",
                "Los fondos apalancados quedan 2 desvíos más vendidos de VIX que en sus últimos 3 años",
                lambda: _primer_dia(z <= -2, z.notna()))
    if "cot_lev_yen" in por_id:
        z = por_id["cot_lev_yen"].reindex(cal)
        agregar("cot_yen_extremo", "Carry trade en yenes extremo", "institucional",
                "Los fondos apalancados quedan 2 desvíos más vendidos de yen que en sus últimos 3 años",
                lambda: _primer_dia(z <= -2, z.notna()))
    if "cot_am_sp" in por_id:
        z = por_id["cot_am_sp"].reindex(cal)
        agregar("cot_am_extremo", "Institucionales muy comprados en el S&P 500", "institucional",
                "Los asset managers quedan 2 desvíos más comprados que en sus últimos 3 años",
                lambda: _primer_dia(z >= 2, z.notna()))
    vent = S.fred("WLCFLPCL")
    if vent.notna().any():
        mediana = vent.rolling(252, min_periods=126).median().shift(1)
        agregar("ventanilla_salto", "Bancos a la ventanilla de la Fed", "institucional",
                "El crédito primario de la Fed se triplica contra su mediana de un año y supera USD 5.000 millones",
                lambda: _primer_dia((vent >= 3 * mediana) & (vent >= 5000), mediana.notna()))
    dep = S.fred("DPSACBW027SBOG")
    if dep.notna().any():
        cambio = dep.pct_change(21, fill_method=None)
        agregar("salida_depositos", "Salida de depósitos", "institucional",
                "Los depósitos de los bancos comerciales caen 1 % o más en un mes",
                lambda: _primer_dia(cambio <= -0.01, cambio.notna()))
    return salida


# --- evaluación ----------------------------------------------------------------------------------

def episodios(marca: pd.Series, refractario: int = REFRACTARIO) -> pd.DatetimeIndex:
    """Primer día de cada racha de marcas separada por al menos `refractario` ruedas sin marcas."""
    m = marca.fillna(0).to_numpy() > 0
    idx = np.flatnonzero(m)
    inicios, ultimo = [], -10 ** 9
    for i in idx:
        if i - ultimo > refractario:
            inicios.append(i)
        ultimo = i
    return marca.index[inicios]


def _binomial(k: int, n: int, p: float) -> float:
    if n == 0 or not 0 < p < 1:
        return np.nan
    return float(binomtest(int(k), int(n), p).pvalue)


def evaluar(detectores: list[Detector], spy: pd.Series, eps: pd.DataFrame) -> pd.DataFrame:
    """Una fila por detector con su frecuencia, poder de anticipación, cobertura de caídas y retornos."""
    cal = spy.index
    y3 = ponderacion.objetivos(spy)["y3"]
    r21 = spy.shift(-21) / spy - 1
    r63 = spy.shift(-63) / spy - 1
    cerca = spy >= CERCA_MAXIMO * spy.rolling(63, min_periods=1).max()
    picos = posiciones(eps, cal, "pico") if not eps.empty else np.array([], dtype=int)
    valles = posiciones(eps, cal, "valle") if not eps.empty else np.array([], dtype=int)
    filas = []
    for d in detectores:
        s = d.serie.reindex(cal)
        valido = s.notna()
        if valido.sum() < 252:
            continue
        conocido = valido & y3.notna()
        base = float(y3[conocido].mean())
        base_calma = float(y3[conocido & cerca].mean())
        ep = episodios(s)
        ep_y = ep[y3.reindex(ep).notna().to_numpy()]
        k = int(y3.reindex(ep_y).sum())
        n = len(ep_y)
        ep_c = ep_y[cerca.reindex(ep_y).to_numpy()]
        k_c = int(y3.reindex(ep_c).sum())
        anios = valido.sum() / 252
        arr = s.fillna(0).to_numpy()
        # Cobertura de los tramos de 5 %: marcas en las 21 ruedas previas al pico y durante la caída.
        antes, durante, anticip, tramos = [], [], [], 0
        for p, v in zip(picos, valles, strict=True):
            if p < VENTANA_ANTES or v < 0 or not valido.iloc[p]:
                continue
            tramos += 1
            ventana = arr[p - VENTANA_ANTES: p + 1]
            antes.append(ventana.max() > 0)
            if ventana.max() > 0:
                anticip.append(VENTANA_ANTES - int(np.argmax(ventana > 0)))
            durante.append(arr[p + 1: v + 1].max() > 0 if v > p else False)
        azar = float(s.rolling(VENTANA_ANTES + 1, min_periods=VENTANA_ANTES + 1).max().where(valido).mean())
        dentro = np.zeros(len(cal), dtype=bool)
        for p, v in zip(picos, valles, strict=True):
            if p >= 0 and v > p:
                dentro[p + 1: v + 1] = True
        dentro_s = pd.Series(dentro, index=cal)
        tasa = float(s[valido].mean())
        tasa_dentro = float(s[valido & dentro_s].mean()) if (valido & dentro_s).any() else np.nan
        marcados = s.index[s.fillna(0) > 0]
        filas.append({
            "id": d.id, "nombre": d.nombre, "familia": d.familia, "descripcion": d.descripcion,
            "desde": s.first_valid_index(), "dias": int((s > 0).sum()), "episodios": len(ep),
            "por_anio": len(ep) / anios if anios else np.nan,
            "base": base, "p_caida": k / n if n else np.nan, "n": n,
            "lift": (k / n) / base if n and base else np.nan, "p": _binomial(k, n, base),
            "base_calma": base_calma, "p_caida_calma": k_c / len(ep_c) if len(ep_c) else np.nan, "n_calma": len(ep_c),
            "lift_calma": (k_c / len(ep_c)) / base_calma if len(ep_c) and base_calma else np.nan,
            "p_calma": _binomial(k_c, len(ep_c), base_calma),
            "tramos": tramos, "cobertura_antes": float(np.mean(antes)) if antes else np.nan, "azar_antes": azar,
            "anticipacion": float(np.median(anticip)) if anticip else np.nan,
            "cobertura_durante": float(np.mean(durante)) if durante else np.nan,
            "concentracion": tasa_dentro / tasa if tasa else np.nan,
            "ret21": float(r21.reindex(ep).median()) if len(ep) else np.nan,
            "ret21_base": float(r21[valido].median()),
            "ret63": float(r63.reindex(ep).median()) if len(ep) else np.nan,
            "ret63_base": float(r63[valido].median()),
            "ultima": marcados.max() if len(marcados) else pd.NaT,
            "activa": bool(len(marcados) and marcados.max() >= cal[-min(RUEDAS_ACTIVA, len(cal))]),
        })
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["q"] = benjamini_hochberg(df["p"])
    df["q_calma"] = benjamini_hochberg(df["p_calma"])
    df["veredicto"] = [veredicto(f) for f in df.to_dict("records")]
    return df.set_index("id")


def veredicto(f: dict) -> str:
    if f["n"] < MIN_EPISODIOS:
        return "Pocos casos para juzgar"
    sig = f["q"] < Q_SIGNIFICATIVO
    sig_c = f["q_calma"] < Q_SIGNIFICATIVO and f["n_calma"] >= MIN_EPISODIOS
    if sig_c and f["lift_calma"] >= LIFT_MINIMO:
        return "Anticipa caídas, también desde máximos"
    if sig and f["lift"] >= LIFT_MINIMO:
        return "Aparece con el estrés ya en curso"
    if sig and f["lift"] <= 1 / LIFT_MINIMO:
        return "Contraria: después hay menos caídas"
    if f["lift"] >= LIFT_MINIMO:
        return "Sugiere riesgo, sin significancia"
    return "No anticipa caídas"


def conteo(detectores: list[Detector], ids: list[str] | None = None, ventana: int = VENTANA_CONTEO) -> pd.Series:
    """Cantidad de detectores distintos que marcaron en las últimas `ventana` ruedas."""
    sel = [d for d in detectores if ids is None or d.id in ids]
    if not sel:
        return pd.Series(dtype=float)
    M = pd.concat([d.serie.fillna(0).rolling(ventana, min_periods=1).max() for d in sel], axis=1)
    return M.sum(axis=1)


def tabla_conteo(c: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """Qué pasó después según cuántas anomalías distintas hubo en las últimas 10 ruedas."""
    Y = ponderacion.objetivos(spy.reindex(c.index))
    df = pd.DataFrame({"c": c, "y3": Y["y3"], "r21": spy.shift(-21) / spy - 1}).dropna()
    if df.empty:
        return pd.DataFrame()
    df["grupo"] = pd.cut(df["c"], [-0.5, 0.5, 1.5, 2.5, np.inf], labels=["0", "1", "2", "3 o más"])
    t = df.groupby("grupo", observed=False).agg(ruedas=("c", "size"), caida=("y3", "mean"), ret21=("r21", "median"))
    t["porcentaje"] = t["ruedas"] / t["ruedas"].sum()
    return t


# --- anomalías de calendario y factores -----------------------------------------------------------

def _condiciones(fechas: pd.DatetimeIndex) -> dict[str, tuple[str, np.ndarray]]:
    mes = fechas.to_period("M")
    s = pd.Series(np.arange(len(fechas)), index=fechas)
    desde_inicio = s.groupby(mes).cumcount().to_numpy()
    desde_fin = s[::-1].groupby(mes[::-1]).cumcount()[::-1].to_numpy()
    siguiente = np.r_[fechas[1:].to_numpy(dtype="datetime64[D]"), np.datetime64("NaT")]
    hoy = fechas.to_numpy(dtype="datetime64[D]")
    pre_feriado = np.zeros(len(fechas), dtype=bool)
    ok = ~np.isnat(siguiente)
    pre_feriado[ok] = np.busday_count(hoy[ok] + 1, siguiente[ok]) > 0
    diciembre, enero = fechas.month == 12, fechas.month == 1
    return {
        "cambio_de_mes": ("Cambio de mes (último día y 3 primeros)", (desde_fin == 0) | (desde_inicio < 3)),
        "lunes": ("Lunes", fechas.dayofweek == 0),
        "pre_feriado": ("Rueda previa a un feriado", pre_feriado),
        "halloween": ("Noviembre a abril (\"sell in May\")", np.isin(fechas.month, [11, 12, 1, 2, 3, 4])),
        "fin_de_anio": ("Rally de fin de año (últimas 5 de diciembre y 2 primeras de enero)",
                        (diciembre & (desde_fin < 5)) | (enero & (desde_inicio < 2))),
        "septiembre": ("Septiembre", fechas.month == 9),
    }


def _t(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 20 or len(b) < 20:
        return np.nan
    return float(ttest_ind(a, b, equal_var=False).statistic)


def _vigencia(t_total: float, t_ultima: float) -> str:
    if np.isnan(t_total) or np.isnan(t_ultima):
        return ""
    if abs(t_total) < 2:
        return "Nunca fue significativa"
    if abs(t_ultima) >= 2 and np.sign(t_ultima) == np.sign(t_total):
        return "Sigue vigente"
    if abs(t_ultima) >= 2:
        return "Se dio vuelta"
    if np.sign(t_ultima) == np.sign(t_total):
        return "Se debilitó (mismo signo, sin significancia)"
    return "Desapareció"


def efectos_calendario(r: pd.Series) -> pd.DataFrame:
    """Retorno diario medio en los días del efecto contra el resto, por era (en puntos básicos)."""
    r = r.dropna()
    if len(r) < 1000:
        return pd.DataFrame()
    cond = _condiciones(r.index)
    filas = []
    for clave, (nombre, m) in cond.items():
        m = pd.Series(m, index=r.index)
        if m.sum() < 50:     # sin días del efecto (p. ej. datos sin feriados)
            continue
        fila = {"id": clave, "efecto": nombre}
        for era, (a, b) in {"Todo": (r.index[0], None), **ERAS}.items():
            rr, mm = r.loc[a:b], m.loc[a:b]
            if len(rr) < 500:
                continue
            dentro, fuera = rr[mm].to_numpy(), rr[~mm].to_numpy()
            fila[f"dif_{era}"] = (dentro.mean() - fuera.mean()) * 1e4 if len(dentro) and len(fuera) else np.nan
            fila[f"t_{era}"] = _t(dentro, fuera)
            if era == "Todo":
                fila["dias_pct"] = float(mm.mean())
                fila["dentro_bps"] = float(dentro.mean() * 1e4)
                fila["fuera_bps"] = float(fuera.mean() * 1e4)
        ultima = list(ERAS)[-1]
        fila["vigencia"] = _vigencia(fila.get("t_Todo", np.nan), fila.get(f"t_{ultima}", np.nan))
        filas.append(fila)
    return pd.DataFrame(filas).set_index("id")


def _max_caida(r: pd.Series) -> float:
    nivel = (1 + r).cumprod()
    return float((nivel / nivel.cummax() - 1).min())


def factores(fac: pd.DataFrame, mom: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prima anual, t y peor caída de tamaño, valor y momentum por era; más el efecto enero en tamaño."""
    series = {}
    if "SMB" in fac:
        series["Tamaño (chicas − grandes)"] = fac["SMB"]
    if "HML" in fac:
        series["Valor (baratas − caras)"] = fac["HML"]
    if mom is not None and not mom.empty:
        series["Momentum (ganadoras − perdedoras)"] = mom.iloc[:, 0]
    filas = []
    for nombre, s in series.items():
        s = s.dropna()
        fila = {"factor": nombre}
        for era, (a, b) in {"Todo": (s.index[0], None), **ERAS}.items():
            ss = s.loc[a:b]
            if len(ss) < 500:
                continue
            fila[f"anual_{era}"] = float(ss.mean() * 252)
            fila[f"t_{era}"] = float(ss.mean() / (ss.std() / np.sqrt(len(ss)))) if ss.std() > 0 else np.nan
            fila[f"caida_{era}"] = _max_caida(ss)
        fila["vigencia"] = _vigencia(fila.get("t_Todo", np.nan), fila.get(f"t_{list(ERAS)[-1]}", np.nan))
        filas.append(fila)
    if "SMB" in fac:
        s = fac["SMB"].dropna()
        fila = {"factor": "Efecto enero en chicas (SMB enero − resto)"}
        for era, (a, b) in {"Todo": (s.index[0], None), **ERAS}.items():
            ss = s.loc[a:b]
            if len(ss) < 500:
                continue
            ene, resto = ss[ss.index.month == 1], ss[ss.index.month != 1]
            if ss.std() == 0:
                continue
            fila[f"anual_{era}"] = float((ene.mean() - resto.mean()) * 21)   # diferencia en un mes de enero
            fila[f"t_{era}"] = _t(ene.to_numpy(), resto.to_numpy())
        fila["vigencia"] = _vigencia(fila.get("t_Todo", np.nan), fila.get(f"t_{list(ERAS)[-1]}", np.nan))
        filas.append(fila)
    return pd.DataFrame(filas).set_index("factor") if filas else pd.DataFrame()


def retornos_largos(S: Series) -> pd.Series:
    """Mercado de EE. UU. desde 1926 (Ken French, con dividendos) y el S&P 500 después de su último dato."""
    fac = S.french("F-F_Research_Data_Factors_daily")
    r = (fac["Mkt-RF"] + fac["RF"]).dropna() if {"Mkt-RF", "RF"} <= set(fac.columns) else pd.Series(dtype=float)
    gspc = S.precio("^GSPC").pct_change(fill_method=None).dropna()
    if r.empty:
        return gspc
    return pd.concat([r, gspc[gspc.index > r.index.max()]]).sort_index()


# --- todo junto ----------------------------------------------------------------------------------

@dataclass
class Anomalias:
    fecha: pd.Timestamp
    detectores: list[Detector]
    evaluacion: pd.DataFrame
    conteo: pd.Series
    conteo_tabla: pd.DataFrame
    conteo_auc: float
    calendario: pd.DataFrame
    factores: pd.DataFrame
    inicio_calendario: pd.Timestamp | None = None
    activas: pd.DataFrame = field(default_factory=pd.DataFrame)


def analizar(S: Series, precios: pd.DataFrame, eps: pd.DataFrame, indicadores: list | None = None) -> Anomalias | None:
    spy = precios["SPY"]
    det = detectar(S, precios, indicadores)
    if not det:
        return None
    ev = evaluar(det, spy, eps)
    c = conteo(det)
    y3 = ponderacion.objetivos(spy.reindex(c.index))["y3"]
    auc_c = ponderacion.auc(c, y3) if len(c) else np.nan
    r_largo = retornos_largos(S)
    cal_t = efectos_calendario(r_largo)
    fac = factores(S.french("F-F_Research_Data_Factors_daily"), S.french("F-F_Momentum_Factor_daily"))
    activas = ev[ev["activa"]] if not ev.empty else pd.DataFrame()
    fecha = spy.dropna().index.max()
    log.info("Anomalías: %d detectores, %d activas hoy; conteo AUC %.3f", len(det), len(activas), auc_c)
    return Anomalias(fecha=fecha, detectores=det, evaluacion=ev, conteo=c, conteo_tabla=tabla_conteo(c, spy),
                     conteo_auc=auc_c, calendario=cal_t, factores=fac,
                     inicio_calendario=r_largo.index.min() if len(r_largo) else None, activas=activas)
