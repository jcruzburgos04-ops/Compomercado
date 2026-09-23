"""Detección de caídas del mercado: tramos (zigzag), episodios bajo el agua y shocks diarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNAS = [
    "pico",
    "valle",
    "confirmacion",
    "fin",
    "recuperacion",
    "precio_pico",
    "precio_valle",
    "profundidad",
    "dias_caida",
    "dias_recuperacion",
    "en_curso",
]


def _dias_entre(indice: pd.DatetimeIndex, a, b) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(indice.get_loc(b) - indice.get_loc(a))


def _completar(eps: list[dict], precios: pd.Series) -> pd.DataFrame:
    if not eps:
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.DataFrame(eps)
    idx = precios.index
    recs = []
    for _, e in df.iterrows():
        posteriores = precios.loc[precios.index > e["valle"]]
        rec = posteriores.index[posteriores >= e["precio_pico"]]
        recs.append(rec[0] if len(rec) else pd.NaT)
    df["recuperacion"] = recs
    df["profundidad"] = df["precio_valle"] / df["precio_pico"] - 1
    df["dias_caida"] = [_dias_entre(idx, a, b) for a, b in zip(df["pico"], df["valle"], strict=True)]
    df["dias_recuperacion"] = [
        _dias_entre(idx, a, b) for a, b in zip(df["valle"], df["recuperacion"], strict=True)
    ]
    return df.reindex(columns=COLUMNAS)


def tramos_zigzag(precios: pd.Series, umbral: float = 0.10, rebote: float | None = None) -> pd.DataFrame:
    """Tramos bajistas: caída >= `umbral` desde un pico, terminada por un rebote >= `rebote` desde el valle.

    Captura cada pierna de una caída (p. ej. las tres de 2022), a diferencia de `episodios_bajo_agua`.
    - pico: máximo previo a la caída.
    - confirmacion: primer día en que la caída desde el pico alcanza el umbral.
    - valle: mínimo del tramo.
    - fin: día en que el rebote confirma el valle (NaT si sigue en curso).
    """
    rebote = umbral if rebote is None else rebote
    p = precios.dropna()
    if p.empty:
        return pd.DataFrame(columns=COLUMNAS)
    fechas, valores = p.index, p.to_numpy(dtype=float)

    eps: list[dict] = []
    bajando = False
    i_pico = 0
    i_valle = 0
    i_conf = 0
    for i in range(1, len(valores)):
        v = valores[i]
        if not bajando:
            if v > valores[i_pico]:
                i_pico = i
            elif v <= valores[i_pico] * (1 - umbral):
                bajando, i_valle, i_conf = True, i, i
        else:
            if v < valores[i_valle]:
                i_valle = i
            elif v >= valores[i_valle] * (1 + rebote):
                eps.append(
                    {
                        "pico": fechas[i_pico],
                        "valle": fechas[i_valle],
                        "confirmacion": fechas[i_conf],
                        "fin": fechas[i],
                        "precio_pico": valores[i_pico],
                        "precio_valle": valores[i_valle],
                        "en_curso": False,
                    }
                )
                bajando = False
                # El nuevo pico se busca desde el valle: el rebote ya es parte de la suba.
                i_pico = int(i_valle + np.argmax(valores[i_valle : i + 1]))
    if bajando:
        eps.append(
            {
                "pico": fechas[i_pico],
                "valle": fechas[i_valle],
                "confirmacion": fechas[i_conf],
                "fin": pd.NaT,
                "precio_pico": valores[i_pico],
                "precio_valle": valores[i_valle],
                "en_curso": True,
            }
        )
    return _completar(eps, p)


def episodios_bajo_agua(precios: pd.Series, umbral: float = 0.10) -> pd.DataFrame:
    """Episodios pico -> valle -> recuperación del pico, con profundidad >= `umbral`."""
    p = precios.dropna()
    if p.empty:
        return pd.DataFrame(columns=COLUMNAS)
    maximo = p.cummax()
    bajo_agua = p < maximo
    eps: list[dict] = []
    grupos = (~bajo_agua).cumsum()
    for _, tramo in p[bajo_agua].groupby(grupos[bajo_agua]):
        pico_precio = maximo.loc[tramo.index[0]]
        pico_fecha = p.loc[: tramo.index[0]].index[-2] if len(p.loc[: tramo.index[0]]) > 1 else tramo.index[0]
        valle_fecha = tramo.idxmin()
        prof = tramo.min() / pico_precio - 1
        if prof > -umbral:
            continue
        conf = tramo.index[np.argmax(tramo.to_numpy() <= pico_precio * (1 - umbral))]
        en_curso = tramo.index[-1] == p.index[-1]
        eps.append(
            {
                "pico": pico_fecha,
                "valle": valle_fecha,
                "confirmacion": conf,
                "fin": pd.NaT if en_curso else p.index[p.index.get_loc(tramo.index[-1]) + 1],
                "precio_pico": pico_precio,
                "precio_valle": tramo.min(),
                "en_curso": en_curso,
            }
        )
    return _completar(eps, p)


def shocks_diarios(precios: pd.Series, umbral: float = -0.02, sigmas: float = 2.0, ventana: int = 63) -> pd.DataFrame:
    """Ruedas con retorno <= `umbral` o <= -`sigmas` desvíos de la volatilidad previa."""
    r = precios.dropna().pct_change()
    vol_previa = r.rolling(ventana, min_periods=ventana // 2).std().shift(1)
    z = r / vol_previa
    mask = (r <= umbral) | (z <= -sigmas)
    return pd.DataFrame({"retorno": r[mask], "z": z[mask]})


def drawdown(precios: pd.Series) -> pd.Series:
    p = precios.dropna()
    return p / p.cummax() - 1


def huella_macro(eps: pd.DataFrame, niveles: dict[str, pd.Series], precios: dict[str, pd.Series],
                 maximos: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Qué hicieron las variables macro entre el pico y el valle de cada episodio.

    - `niveles`: series en nivel (tasas, spreads) -> diferencia valle - pico, con prefijo `d_`.
    - `precios`: series de precio (dólar, petróleo, yen) -> variación %, con prefijo `r_`.
    - `maximos`: series de las que interesa el máximo en el tramo (VIX), con sufijo `_max`.
    """
    maximos = maximos or {}
    filas = []
    for _, e in eps.iterrows():
        fila: dict[str, float] = {}
        for nombre, s in niveles.items():
            a, b = _valor_en(s, e["pico"]), _valor_en(s, e["valle"])
            fila[f"d_{nombre}"] = b - a
        for nombre, s in precios.items():
            a, b = _valor_en(s, e["pico"]), _valor_en(s, e["valle"])
            fila[f"r_{nombre}"] = b / a - 1 if a and not np.isnan(a) else np.nan
        for nombre, s in maximos.items():
            tramo = s.loc[e["pico"] : e["valle"]].dropna()
            fila[f"{nombre}_max"] = float(tramo.max()) if len(tramo) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas, index=eps.index)


def _valor_en(s: pd.Series, fecha) -> float:
    s = s.dropna()
    s = s.loc[:fecha]
    return float(s.iloc[-1]) if len(s) else np.nan


def clasificar(eps: pd.DataFrame, reglas: list[dict]) -> pd.Series:
    """Asigna un tipo de caída por episodio evaluando reglas en orden (config/analisis.yaml)."""
    tipos = []
    for _, fila in eps.iterrows():
        entorno = fila.to_dict()
        tipo = "Sin clasificar"
        for regla in reglas:
            try:
                if eval(regla["condicion"], {"__builtins__": {}}, entorno):  # noqa: S307 - config local
                    tipo = regla["tipo"]
                    break
            except (NameError, TypeError):
                continue
        tipos.append(tipo)
    return pd.Series(tipos, index=eps.index, name="tipo")
