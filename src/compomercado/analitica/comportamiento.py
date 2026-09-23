"""Mapa de comportamiento: cómo se mueve cada activo frente a una referencia, en general y en cada caída."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_OBS = 120


def retornos(precios: pd.DataFrame | pd.Series, semanal: bool = False):
    if semanal:
        precios = precios.resample("W-FRI").last()
    return precios / precios.shift(1) - 1


def _media_geometrica(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    return float(np.prod(1 + r.to_numpy()) ** (1 / len(r)) - 1)


def metricas_historicas(ra: pd.Series, rm: pd.Series, periodos_anio: int = 252) -> dict[str, float]:
    """Beta total/bajista/alcista, captura, correlación en estrés, MES y co-excedencia de cola."""
    df = pd.concat([ra, rm], axis=1, keys=["a", "m"]).dropna()
    n = len(df)
    salida: dict[str, float] = {"n_obs": float(n)}
    if n < MIN_OBS * periodos_anio / 252:
        return salida
    a, m = df["a"], df["m"]
    salida["beta"] = a.cov(m) / m.var()
    abajo, arriba = m < 0, m > 0
    salida["beta_bajista"] = a[abajo].cov(m[abajo]) / m[abajo].var()
    salida["beta_alcista"] = a[arriba].cov(m[arriba]) / m[arriba].var()
    salida["asimetria"] = salida["beta_bajista"] - salida["beta_alcista"]
    salida["corr"] = a.corr(m)
    estres = m <= m.quantile(0.10)
    salida["corr_estres"] = a[estres].corr(m[estres])
    cola = m <= m.quantile(0.05)
    salida["mes"] = a[cola].mean()
    salida["beta_cola"] = a[cola].mean() / m[cola].mean()
    salida["coexcedencia"] = float((a[cola] <= a.quantile(0.05)).mean())
    salida["vol"] = a.std() * np.sqrt(periodos_anio)

    mensual = (1 + df).resample("ME").prod() - 1
    mensual = mensual[(df.resample("ME").size() > 0)]
    baj, alc = mensual["m"] < 0, mensual["m"] > 0
    gm_b, gm_a = _media_geometrica(mensual["m"][baj]), _media_geometrica(mensual["m"][alc])
    salida["captura_bajista"] = _media_geometrica(mensual["a"][baj]) / gm_b if gm_b else np.nan
    salida["captura_alcista"] = _media_geometrica(mensual["a"][alc]) / gm_a if gm_a else np.nan
    return salida


def metricas_por_episodio(
    pa: pd.Series,
    pm: pd.Series,
    eps: pd.DataFrame,
    pre: int = 63,
    rebotes: tuple[int, ...] = (21, 63),
    margen_piso: int = 21,
) -> pd.DataFrame:
    """Una fila por episodio: retorno del activo, relativo, drawdown propio, lead-lag y rebote.

    `pa` y `pm` deben estar en el mismo calendario (el de la referencia).
    - lag_techo: ruedas entre el máximo del activo (en [pico-pre, valle], sin entrar en el tramo
      anterior) y el pico del mercado. Negativo = el activo hizo techo antes.
    - lag_piso: ruedas entre el mínimo del activo (en [pico, valle+margen], sin entrar en el tramo
      siguiente) y el valle del mercado. Negativo = el activo tocó fondo antes.
    """
    idx = pm.index
    pa = pa.reindex(idx)
    n = len(idx)
    eps = eps.sort_values("pico")
    picos = [idx.get_loc(p) for p in eps["pico"]]
    valles = [idx.get_loc(v) for v in eps["valle"]]
    filas = []
    for k, id_ep in enumerate(eps.index):
        fila: dict[str, float] = {"episodio": id_ep}
        i_p, i_v = picos[k], valles[k]
        # Las ventanas de techo y piso no se meten en el tramo anterior ni en el siguiente.
        desde = max(0, i_p - pre, valles[k - 1] + 1 if k > 0 else 0)
        hasta = min(n, i_v + margen_piso + 1, picos[k + 1] + 1 if k + 1 < len(picos) else n)
        a_p, a_v = pa.iloc[i_p], pa.iloc[i_v]
        m_p, m_v = pm.iloc[i_p], pm.iloc[i_v]
        ret_m = m_v / m_p - 1
        fila["ret_mercado"] = ret_m
        if np.isnan(a_p) or np.isnan(a_v):
            filas.append(fila)
            continue
        ret_a = a_v / a_p - 1
        fila["ret_activo"] = ret_a
        fila["relativo"] = ret_a - ret_m
        fila["captura"] = ret_a / ret_m if ret_m else np.nan
        tramo = pa.iloc[i_p : i_v + 1]
        fila["dd_propio"] = float((tramo / tramo.cummax() - 1).min())

        previo = pa.iloc[desde : i_v + 1]
        if previo.notna().any():
            fila["lag_techo"] = float(idx.get_loc(previo.idxmax()) - i_p)
        piso = pa.iloc[i_p:hasta]
        if piso.notna().any():
            fila["lag_piso"] = float(idx.get_loc(piso.idxmin()) - i_v)

        for h in rebotes:
            if i_v + h < n and not np.isnan(pa.iloc[i_v + h]):
                reb_a = pa.iloc[i_v + h] / a_v - 1
                reb_m = pm.iloc[i_v + h] / m_v - 1
                fila[f"rebote_{h}"] = reb_a
                fila[f"rebote_rel_{h}"] = reb_a - reb_m
        if i_p - pre >= 0 and not np.isnan(pa.iloc[i_p - pre]):
            fila["pre_pico_rel"] = (a_p / pa.iloc[i_p - pre] - 1) - (m_p / pm.iloc[i_p - pre] - 1)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("episodio")


def resumen_episodios(tabla: pd.DataFrame) -> dict[str, float]:
    """Agregados robustos (medianas y tasa de acierto) de la tabla por episodio de un activo."""
    t = tabla.dropna(subset=["relativo"]) if "relativo" in tabla else tabla.iloc[0:0]
    salida: dict[str, float] = {"n_episodios": float(len(t))}
    if t.empty:
        return salida
    salida["captura_mediana"] = t["captura"].median()
    salida["relativo_mediano"] = t["relativo"].median()
    salida["acierto_defensivo"] = float((t["relativo"] > 0).mean())
    salida["dd_propio_mediano"] = t["dd_propio"].median()
    for col in ("lag_techo", "lag_piso", "rebote_rel_21", "rebote_rel_63", "pre_pico_rel"):
        if col in t:
            salida[f"{col}_mediano"] = t[col].median()
    return salida


def fuerza_relativa(precios: pd.DataFrame, pm: pd.Series, horizontes=(21, 63, 126)) -> pd.DataFrame:
    """Retorno relativo contra la referencia y su percentil dentro del universo."""
    salida = {}
    for h in horizontes:
        if len(pm.dropna()) <= h:
            continue
        rm = pm.iloc[-1] / pm.iloc[-1 - h] - 1
        ult = precios.iloc[-1]
        prev = precios.iloc[-1 - h]
        rel = (ult / prev - 1) - rm
        salida[f"rs_{h}"] = rel
        salida[f"rs_{h}_pct"] = rel.rank(pct=True) * 100
    return pd.DataFrame(salida)


def _rango(s: pd.Series, mayor_es_mejor: bool) -> pd.Series:
    r = s.rank(pct=True) * 100
    return r if mayor_es_mejor else 100 - r + (100 / max(s.notna().sum(), 1))


def puntajes(tabla: pd.DataFrame, min_episodios: int = 3) -> pd.DataFrame:
    """Puntaje Refugio (qué aguanta) y Rebote (qué lidera la salida), 0-100 dentro del universo."""
    t = tabla.copy()
    pocos = t.get("n_episodios", pd.Series(0, index=t.index)) < min_episodios
    ep = lambda c: t[c].mask(pocos) if c in t else pd.Series(np.nan, index=t.index)  # noqa: E731
    col = lambda c: t[c] if c in t else pd.Series(np.nan, index=t.index)  # noqa: E731

    refugio = pd.concat(
        [
            _rango(ep("captura_mediana"), False),
            _rango(ep("acierto_defensivo"), True),
            _rango(col("beta_bajista"), False),
            _rango(col("beta_cola"), False),
            _rango(col("corr_estres"), False),
        ],
        axis=1,
    )
    rebote = pd.concat(
        [
            _rango(ep("rebote_rel_63_mediano"), True),
            _rango(ep("rebote_rel_21_mediano"), True),
            _rango(col("beta_alcista"), True),
            _rango(ep("lag_piso_mediano"), False),
        ],
        axis=1,
    )
    t["puntaje_refugio"] = refugio.mean(axis=1, skipna=True).where(refugio.notna().sum(axis=1) >= 3)
    t["puntaje_rebote"] = rebote.mean(axis=1, skipna=True).where(rebote.notna().sum(axis=1) >= 2)
    return t


def mapa(
    precios: pd.DataFrame,
    ref: str | pd.Series,
    eps: pd.DataFrame,
    asincronicos: set[str] | None = None,
    ventana_reciente: int = 756,
    pre: int = 63,
    rebotes: tuple[int, ...] = (21, 63),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye el mapa de comportamiento de todos los activos de `precios` contra `ref`.

    `precios` debe estar alineado al calendario de la referencia. Devuelve (tabla por activo,
    tabla larga por activo y episodio).
    """
    asincronicos = asincronicos or set()
    pm = precios[ref] if isinstance(ref, str) else ref
    pm = pm.dropna()
    precios = precios.reindex(pm.index)
    rm_d = retornos(pm)
    rm_s = retornos(pm, semanal=True)

    filas, largas = {}, []
    for t in precios.columns:
        pa = precios[t]
        if pa.dropna().empty:
            continue
        semanal = t in asincronicos
        ra = retornos(pa, semanal=semanal)
        rm = rm_s if semanal else rm_d
        anual = 52 if semanal else 252
        total = metricas_historicas(ra, rm, anual)
        reciente_n = ventana_reciente // 5 if semanal else ventana_reciente
        reciente = metricas_historicas(ra.iloc[-reciente_n:], rm.iloc[-reciente_n:], anual)
        fila = dict(total)
        fila.update({f"{k}_3a": v for k, v in reciente.items() if k in ("beta", "beta_bajista", "corr", "corr_estres", "captura_bajista", "captura_alcista")})
        fila["frecuencia"] = "semanal" if semanal else "diaria"
        fila["inicio"] = pa.first_valid_index()

        por_ep = metricas_por_episodio(pa, pm, eps, pre=pre, rebotes=rebotes)
        fila.update(resumen_episodios(por_ep))
        por_ep["activo"] = t
        largas.append(por_ep.reset_index())
        filas[t] = fila

    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla = tabla.join(fuerza_relativa(precios[list(filas)].ffill(), pm))
    tabla = puntajes(tabla)
    larga = pd.concat(largas, ignore_index=True) if largas else pd.DataFrame()
    return tabla, larga


def por_tipo(larga: pd.DataFrame, tipos: pd.Series, min_n: int = 2) -> pd.DataFrame:
    """Retorno relativo mediano de cada activo según el tipo de caída (tipos con >= min_n episodios)."""
    if larga.empty:
        return pd.DataFrame()
    df = larga.merge(tipos.rename("tipo"), left_on="episodio", right_index=True, how="left")
    df = df.dropna(subset=["relativo"])
    conteo = df.groupby("tipo")["episodio"].nunique()
    validos = conteo[conteo >= min_n].index
    df = df[df["tipo"].isin(validos)]
    tabla = df.pivot_table(index="activo", columns="tipo", values="relativo", aggfunc="median")
    tabla.columns = [f"{c} (n={conteo[c]})" for c in tabla.columns]
    return tabla
