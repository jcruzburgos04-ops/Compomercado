"""Huellas de las caídas: cómo estaban los indicadores antes, durante y al final de cada tramo bajista
de SPY, y qué días del pasado se parecen al de hoy.

Tres preguntas:

1. **¿Qué tenían en común las caídas?** Para cada tramo se toma el percentil de riesgo de cada
   indicador y de cada pilar en cinco momentos: un mes y una semana antes del pico, el pico, el día en
   que la caída alcanza el umbral (confirmación) y el valle. Se compara contra todos los días de la
   historia: media, porcentaje de casos en zona alta (≥ 70) y significancia (z de la media, con la
   corrección de Benjamini-Hochberg porque se comparan muchos indicadores a la vez).
2. **¿Cómo evolucionan alrededor del pico y del valle?** Mediana de cada pilar entre −63 y +63 ruedas.
3. **¿A qué días del pasado se parece hoy?** Vecinos más cercanos en el espacio de los pilares. La
   probabilidad por análogos es la fracción de esos vecinos que antecedió una caída ≥ 5 % en 21
   ruedas. Es walk-forward: cada día solo mira vecinos cuyo resultado ya se conocía (purga de 21
   ruedas), así que se evalúa fuera de muestra igual que la ponderación.

El pico se conoce después: por construcción es un máximo local, así que los indicadores de tendencia
se ven tranquilos ahí. Lo que sirve para anticipar son las semanas previas y, sobre todo, la
evaluación fuera de muestra de los análogos.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..indicadores import ponderacion
from ..indicadores.estado import caida_maxima_futura, percentil_expansivo

log = logging.getLogger(__name__)

# momento -> (columna del tramo que sirve de ancla, desplazamiento en ruedas, etiqueta)
MOMENTOS = {
    "pre21": ("pico", -21, "1 mes antes del pico"),
    "pre5": ("pico", -5, "1 semana antes del pico"),
    "pico": ("pico", 0, "Pico"),
    "conf": ("confirmacion", 0, "Confirmación"),
    "valle": ("valle", 0, "Valle"),
}
ZONA_ALTA = 70.0
ZONA_BAJA = 30.0
Q_SIGNIFICATIVO = 0.10
MIN_TRAMOS = 8
PREFIJO_PILAR = "P:"

K_VECINOS = 50
PURGA = 21                # el objetivo mira 21 ruedas adelante
MIN_PASADO = 504          # días con resultado conocido para empezar a estimar
MIN_DIMS = 6              # pilares en común para comparar dos días
CAMBIO = 21               # el vector de estado incluye el cambio del total en 21 ruedas
SEPARACION = 63           # ruedas mínimas entre dos análogos que se muestran
EXCLUIR_RECIENTES = 126   # para listar análogos históricos: se ignora el último semestre
PRIMER_ANIO = 2005        # igual que la ponderación, para comparar en los mismos años


# --- instantáneas por momento --------------------------------------------------------------------

def posiciones(eps: pd.DataFrame, indice: pd.DatetimeIndex, ancla: str, desplazamiento: int = 0) -> np.ndarray:
    """Posición en `indice` de cada tramo (ancla + desplazamiento); -1 si cae fuera."""
    fechas = pd.DatetimeIndex(eps[ancla])
    pos = indice.searchsorted(fechas)
    pos = np.where(fechas.isna(), -1, pos)
    pos = np.where(pos >= 0, pos + desplazamiento, -1)
    return np.where((pos >= 0) & (pos < len(indice)), pos, -1)


def instantaneas(eps: pd.DataFrame, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """momento -> DataFrame tramo x columna con el valor de X en ese momento de cada tramo."""
    arr = X.to_numpy(dtype=float)
    salida = {}
    for m, (ancla, desp, _) in MOMENTOS.items():
        pos = posiciones(eps, X.index, ancla, desp)
        valores = np.full((len(eps), X.shape[1]), np.nan)
        ok = pos >= 0
        valores[ok] = arr[pos[ok]]
        salida[m] = pd.DataFrame(valores, index=eps.index, columns=X.columns)
    return salida


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """q-valores (tasa de falsos descubrimientos) para una serie de p-valores; NaN se respetan."""
    v = p.dropna()
    if v.empty:
        return p * np.nan
    orden = v.sort_values()
    m = len(orden)
    q = orden.to_numpy() * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
    return pd.Series(q, index=orden.index).reindex(p.index)


def comparar(snaps: dict[str, pd.DataFrame], X: pd.DataFrame) -> pd.DataFrame:
    """Cada columna de X en cada momento de los tramos contra todos los días con dato."""
    filas = []
    for col in X.columns:
        base = X[col].dropna()
        if len(base) < MIN_PASADO:
            continue
        mb, sb = float(base.mean()), float(base.std())
        alta_b, baja_b = float((base >= ZONA_ALTA).mean()), float((base <= ZONA_BAJA).mean())
        for m in MOMENTOS:
            v = snaps[m][col].dropna()
            n = len(v)
            fila = {"columna": col, "momento": m, "n": n, "media_base": mb, "alta_base": alta_b, "baja_base": baja_b}
            if n:
                fila.update(media=float(v.mean()), alta=float((v >= ZONA_ALTA).mean()),
                            baja=float((v <= ZONA_BAJA).mean()))
            if n >= MIN_TRAMOS and sb > 0:
                z = (fila["media"] - mb) / (sb / np.sqrt(n))
                fila.update(z=float(z), p=float(2 * norm.sf(abs(z))))
            filas.append(fila)
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    for c in ("media", "alta", "baja", "z", "p"):
        if c not in df:
            df[c] = np.nan
    df["dif"] = df["media"] - df["media_base"]
    df["lift_alta"] = df["alta"] / df["alta_base"].replace(0, np.nan)
    df["q"] = benjamini_hochberg(df["p"])
    df["significativo"] = df["q"] < Q_SIGNIFICATIVO
    return df


def lectura(comp: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna: media por momento y qué papel jugó en las caídas."""
    if comp.empty:
        return pd.DataFrame()
    media = comp.pivot(index="columna", columns="momento", values="media")
    dif = comp.pivot(index="columna", columns="momento", values="dif")
    sig = comp.pivot(index="columna", columns="momento", values="significativo").fillna(False).astype(bool)
    q = comp.pivot(index="columna", columns="momento", values="q")
    alta = comp.pivot(index="columna", columns="momento", values="alta")
    lift = comp.pivot(index="columna", columns="momento", values="lift_alta")
    n = comp.pivot(index="columna", columns="momento", values="n")
    base = comp.groupby("columna")[["media_base", "alta_base"]].first()

    def papel(c) -> str:
        sube = {m: bool(sig.at[c, m] and dif.at[c, m] > 0) for m in MOMENTOS if m in sig.columns}
        baja = {m: bool(sig.at[c, m] and dif.at[c, m] < 0) for m in MOMENTOS if m in sig.columns}
        if sube.get("pre21") or sube.get("pre5"):
            return "Anticipa: alto antes del pico"
        if baja.get("pre21") or baja.get("pre5"):
            return "Calma previa: bajo antes del pico"
        if sube.get("conf"):
            return "Confirma: sube con la caída"
        if sube.get("valle"):
            return "Marca el piso: máximo en el valle"
        return "Sin patrón claro"

    salida = pd.DataFrame({f"media_{m}": media.get(m) for m in MOMENTOS})
    salida["alta_pre5"] = alta.get("pre5")
    salida["lift_pre5"] = lift.get("pre5")
    salida["alta_conf"] = alta.get("conf")
    salida["lift_conf"] = lift.get("conf")
    salida["q_pre"] = q[[m for m in ("pre21", "pre5") if m in q.columns]].min(axis=1)
    salida["q_conf"] = q.get("conf")
    salida["n"] = n.get("pre5")
    salida = salida.join(base)
    salida["papel"] = [papel(c) for c in salida.index]
    return salida


# --- trayectorias ------------------------------------------------------------------------------

def trayectoria(eps: pd.DataFrame, X: pd.DataFrame, ancla: str, antes: int = 63, despues: int = 63) -> pd.DataFrame:
    """Mediana entre tramos de cada columna, de `antes` ruedas antes a `despues` después del ancla."""
    pos = posiciones(eps, X.index, ancla)
    pos = pos[pos >= 0]
    desp = np.arange(-antes, despues + 1)
    if not len(pos):
        return pd.DataFrame(index=desp, columns=X.columns, dtype=float)
    arr = X.to_numpy(dtype=float)
    idx = pos[:, None] + desp[None, :]
    ok = (idx >= 0) & (idx < len(arr))
    datos = arr[np.clip(idx, 0, len(arr) - 1)]
    datos[~ok] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(datos, axis=0)
    return pd.DataFrame(med, index=pd.Index(desp, name="ruedas"), columns=X.columns)


# --- análogos: vecinos más cercanos walk-forward -------------------------------------------------

def rasgos(pilares: pd.DataFrame, cambio: int = CAMBIO) -> pd.DataFrame:
    """Vector de estado diario: percentil histórico (point-in-time) de cada pilar y del cambio del total.

    Cada pilar se pasa a su propio percentil para que todos pesen igual en la distancia: un pilar con
    muchos indicadores se mueve menos que uno con pocos.
    """
    cols = [c for c in pilares.columns if c != "total"]
    F = pd.DataFrame(index=pilares.index)
    for c in cols:
        F[c] = percentil_expansivo(pilares[c]).reindex(pilares.index) * 100
    if "total" in pilares and cambio:
        F["cambio_total"] = percentil_expansivo(pilares["total"].diff(cambio)).reindex(pilares.index) * 100
    return F


def distancias(A: np.ndarray, B: np.ndarray, min_dims: int = MIN_DIMS) -> np.ndarray:
    """Distancia cuadrática media entre filas de A y de B sobre las dimensiones que ambas tienen."""
    ma, mb = ~np.isnan(A), ~np.isnan(B)
    a0, b0 = np.where(ma, A, 0.0), np.where(mb, B, 0.0)
    mbf, maf = mb.astype(float), ma.astype(float)
    cuenta = maf @ mbf.T
    ss = (a0 ** 2) @ mbf.T + maf @ (b0 ** 2).T - 2 * a0 @ b0.T
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt(np.maximum(ss, 0.0) / cuenta)
    d[cuenta < min_dims] = np.inf
    return d


def probabilidad_analogos(F: pd.DataFrame, y: pd.Series, k: int = K_VECINOS, purga: int = PURGA,
                          min_pasado: int = MIN_PASADO, min_dims: int = MIN_DIMS, desde=None,
                          bloque: int = 400) -> pd.Series:
    """Fracción de los k días pasados más parecidos a cada día que antecedieron el evento `y`.

    Para el día i solo se usan días j <= i - purga: su resultado ya era conocido en i.
    """
    X = F.to_numpy(dtype=float)
    yy = y.reindex(F.index).to_numpy(dtype=float)
    validas = (~np.isnan(X)).sum(axis=1) >= min_dims
    salida = np.full(len(X), np.nan)
    objetivo = np.flatnonzero(validas)
    if desde is not None:
        objetivo = objetivo[F.index[objetivo] >= pd.Timestamp(desde)]
    candidatos_ok = validas & ~np.isnan(yy)
    for inicio in range(0, len(objetivo), bloque):
        filas = objetivo[inicio: inicio + bloque]
        tope = int(filas.max()) - purga
        if tope < 0:
            continue
        cand = np.flatnonzero(candidatos_ok[: tope + 1])
        if len(cand) < max(min_pasado, k):
            continue
        D = distancias(X[filas], X[cand], min_dims)
        D[cand[None, :] > (filas[:, None] - purga)] = np.inf
        disponibles = np.isfinite(D).sum(axis=1)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        dsel = np.take_along_axis(D, idx, axis=1)
        ysel = np.where(np.isfinite(dsel), yy[cand][idx], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prob = np.nanmean(ysel, axis=1)
        prob[disponibles < max(min_pasado, k)] = np.nan
        salida[filas] = prob
    return pd.Series(salida, index=F.index, name="prob_analogos")


def analogos_de(F: pd.DataFrame, fecha, n: int = 5, separacion: int = SEPARACION,
                excluir_recientes: int = EXCLUIR_RECIENTES, min_dims: int = MIN_DIMS) -> pd.DataFrame:
    """Los n días más parecidos a `fecha`, separados entre sí por `separacion` ruedas (uno por episodio)."""
    fecha = pd.Timestamp(fecha)
    i = F.index.get_loc(fecha)
    fin = i - excluir_recientes
    if fin <= 0:
        return pd.DataFrame(columns=["distancia"])
    X = F.to_numpy(dtype=float)
    d = distancias(X[i: i + 1], X[:fin], min_dims)[0]
    elegidos: list[int] = []
    for j in np.argsort(d):
        if not np.isfinite(d[j]):
            break
        if all(abs(j - e) >= separacion for e in elegidos):
            elegidos.append(int(j))
        if len(elegidos) == n:
            break
    return pd.DataFrame({"distancia": d[elegidos]}, index=F.index[elegidos])


def caidas_parecidas(F: pd.DataFrame, fecha, eps: pd.DataFrame, momento: str = "pre5",
                     min_dims: int = MIN_DIMS) -> pd.DataFrame:
    """Distancia entre el estado de `fecha` y el de cada tramo en `momento` (más cerca = más parecido)."""
    ancla, desp, _ = MOMENTOS[momento]
    pos = posiciones(eps, F.index, ancla, desp)
    ok = pos >= 0
    salida = eps.loc[ok].copy()
    if salida.empty:
        salida["distancia"] = []
        return salida
    X = F.to_numpy(dtype=float)
    i = F.index.get_loc(pd.Timestamp(fecha))
    salida["distancia"] = distancias(X[i: i + 1], X[pos[ok]], min_dims)[0]
    salida = salida[np.isfinite(salida["distancia"])]
    # Un tramo que empezó hace menos de un mes es el presente, no un análogo.
    salida = salida[salida["pico"] < F.index[max(0, i - PURGA)]]
    return salida.sort_values("distancia")


def calibracion(prob: pd.Series, y: pd.Series, cortes=(0, 0.05, 0.10, 0.20, 0.35, 1.0)) -> pd.DataFrame:
    """Qué pasó realmente según el nivel de la probabilidad por análogos (fuera de muestra)."""
    df = pd.concat([prob, y], axis=1, keys=["p", "y"]).dropna()
    if df.empty:
        return pd.DataFrame()
    etiquetas = [f"{a:.0%}–{b:.0%}" for a, b in zip(cortes[:-1], cortes[1:], strict=True)]
    df["tramo"] = pd.cut(df["p"], list(cortes), labels=etiquetas, include_lowest=True)
    t = df.groupby("tramo", observed=False).agg(ruedas=("p", "size"), prob_media=("p", "mean"), realizado=("y", "mean"))
    t["porcentaje_ruedas"] = t["ruedas"] / t["ruedas"].sum()
    return t


# --- todo junto ----------------------------------------------------------------------------------

@dataclass
class Huellas:
    tramos: pd.DataFrame                      # tramos de SPY con datos de indicadores
    instantaneas: dict[str, pd.DataFrame]     # momento -> tramo x columna (indicadores y "P:pilar")
    comparacion: pd.DataFrame                 # columna x momento vs todos los días
    lectura: pd.DataFrame                     # columna: medias por momento + papel
    trayectoria_pico: pd.DataFrame
    trayectoria_valle: pd.DataFrame
    por_tipo: pd.DataFrame                    # tipo x (momento, pilar): media
    rasgos: pd.DataFrame
    prob: pd.Series                           # probabilidad por análogos (walk-forward)
    base_y3: float                            # frecuencia de caída ≥5 %/21r en todos los días evaluados
    evaluacion: pd.DataFrame
    calibracion: pd.DataFrame
    analogos_hoy: pd.DataFrame
    caidas_parecidas: pd.DataFrame
    fecha: pd.Timestamp | None = None
    vecinos_hoy: dict = field(default_factory=dict)


def _por_tipo(snaps: dict[str, pd.DataFrame], tramos: pd.DataFrame, pilares: list[str]) -> pd.DataFrame:
    if "tipo" not in tramos:
        return pd.DataFrame()
    partes = {}
    for m in ("pre5", "conf", "valle"):
        s = snaps[m][[PREFIJO_PILAR + p for p in pilares]].copy()
        s.columns = pilares
        partes[m] = s.groupby(tramos["tipo"]).mean()
    t = pd.concat(partes, axis=1)
    t[("", "tramos")] = tramos.groupby("tipo").size()
    return t


def analizar(eps: pd.DataFrame, riesgo: pd.DataFrame, pilares_igual: pd.DataFrame, total_principal: pd.Series,
             spy: pd.Series, primer_anio: int = PRIMER_ANIO) -> Huellas | None:
    """`riesgo`: percentiles de riesgo por indicador; `pilares_igual`: pilares con pesos iguales (existen
    desde antes que los ponderados, que arrancan en 2005)."""
    if eps.empty or pilares_igual.empty:
        return None
    cal = pilares_igual.index
    pil_cols = list(pilares_igual.columns)
    X = pd.concat([riesgo.reindex(cal), pilares_igual.add_prefix(PREFIJO_PILAR)], axis=1)
    snaps = instantaneas(eps, X)
    # Solo cuentan los tramos con el puntaje total disponible una semana antes del pico.
    con_datos = snaps["pre5"][PREFIJO_PILAR + "total"].notna() if PREFIJO_PILAR + "total" in X else pd.Series(False, index=eps.index)
    tramos = eps.loc[con_datos]
    snaps = {m: s.loc[con_datos] for m, s in snaps.items()}
    if len(tramos) < MIN_TRAMOS:
        log.warning("Huellas: solo %d tramos con datos", len(tramos))
    comp = comparar(snaps, X)
    lect = lectura(comp)
    Xp = pilares_igual
    tray_pico = trayectoria(tramos, Xp, "pico")
    tray_valle = trayectoria(tramos, Xp, "valle")
    pilares_sin_total = [c for c in pil_cols if c != "total"] + (["total"] if "total" in pil_cols else [])
    tipo = _por_tipo(snaps, tramos, pilares_sin_total)

    # Análogos.
    Y = ponderacion.objetivos(spy.reindex(cal))
    F = rasgos(pilares_igual)
    desde = pd.Timestamp(f"{primer_anio}-01-01")
    inicio_prob = F.dropna(thresh=MIN_DIMS).index.min()
    prob = probabilidad_analogos(F, Y["y3"], desde=inicio_prob)
    fin = cal.max()
    periodos = {"Todo el período": (desde, fin), "2005–2014": (desde, "2014-12-31"),
                "2015–2019": ("2015-01-01", "2019-12-31"), "2020 en adelante": ("2020-01-01", fin)}
    # Combinado: promedio del total ponderado y del percentil histórico (point-in-time) de los análogos.
    combinado = pd.concat([total_principal, percentil_expansivo(prob) * 100], axis=1).mean(axis=1, skipna=False)
    variantes = {"Análogos (vecinos más cercanos)": prob, "Riesgo total ponderado (principal)": total_principal,
                 "Combinado: ponderado + análogos": combinado,
                 "Riesgo total con pesos iguales": pilares_igual.get("total", pd.Series(dtype=float))}
    ev = ponderacion.evaluar(variantes, Y, periodos)
    cal_t = calibracion(prob.loc[desde:], Y["y3"])
    base = float(Y["y3"].loc[desde:].reindex(prob.loc[desde:].dropna().index).mean())

    hoy = F.dropna(thresh=MIN_DIMS).index.max()
    analogos = pd.DataFrame()
    parecidas = pd.DataFrame()
    vecinos: dict = {}
    if pd.notna(hoy):
        analogos = analogos_de(F, hoy)
        if not analogos.empty:
            p = spy.reindex(cal)
            analogos["caida_max_21"] = caida_maxima_futura(p, 21).reindex(analogos.index)
            analogos["caida_max_63"] = caida_maxima_futura(p, 63).reindex(analogos.index)
            analogos["ret_63"] = (p.shift(-63) / p - 1).reindex(analogos.index)
            analogos["total"] = pilares_igual.get("total", pd.Series(dtype=float)).reindex(analogos.index)
        parecidas = caidas_parecidas(F, hoy, tramos)
        vecinos = {"prob": float(prob.get(hoy, np.nan)), "k": K_VECINOS}
    log.info("Huellas: %d tramos con datos; análogos hoy %.0f %% (base %.0f %%)", len(tramos),
             100 * vecinos.get("prob", np.nan), 100 * base)
    return Huellas(tramos=tramos, instantaneas=snaps, comparacion=comp, lectura=lect, trayectoria_pico=tray_pico,
                   trayectoria_valle=tray_valle, por_tipo=tipo, rasgos=F, prob=prob, base_y3=base, evaluacion=ev,
                   calibracion=cal_t, analogos_hoy=analogos, caidas_parecidas=parecidas, fecha=hoy, vecinos_hoy=vecinos)
