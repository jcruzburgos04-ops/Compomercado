"""Snapshot diario de cadenas de opciones (Yahoo) para construir historia propia de put/call, skew y GEX.

Yahoo solo muestra la cadena de hoy, así que la historia se acumula en el registro forward día a día.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TASA = 0.04  # tasa libre de riesgo aproximada; el GEX es poco sensible a este valor


def gamma_bs(S: float, K: np.ndarray, T: np.ndarray, iv: np.ndarray, r: float = TASA) -> np.ndarray:
    iv = np.clip(iv, 1e-4, None)
    T = np.clip(T, 1 / 365, None)
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) / (S * iv * np.sqrt(T))


def resumir_cadena(cadena: pd.DataFrame, spot: float, hoy: pd.Timestamp) -> dict[str, float]:
    """Métricas agregadas de una cadena con columnas: tipo (call/put), vencimiento, strike,
    volume, openInterest, impliedVolatility."""
    c = cadena.copy()
    c["volume"] = c["volume"].fillna(0)
    c["openInterest"] = c["openInterest"].fillna(0)
    c = c[(c["impliedVolatility"] > 0.01) & (c["impliedVolatility"] < 5)]
    T = (pd.to_datetime(c["vencimiento"]) - hoy).dt.days.clip(lower=1).to_numpy() / 365
    c["gamma"] = gamma_bs(spot, c["strike"].to_numpy(), T, c["impliedVolatility"].to_numpy())
    signo = np.where(c["tipo"] == "call", 1.0, -1.0)
    # Dólares de delta que los dealers tendrían que operar por un movimiento de 1 % (convención estándar:
    # dealers comprados en calls y vendidos en puts).
    gex = float((signo * c["gamma"] * c["openInterest"] * 100 * spot**2 * 0.01).sum())

    calls, puts = c[c["tipo"] == "call"], c[c["tipo"] == "put"]
    salida = {
        "spot": spot,
        "pc_volumen": puts["volume"].sum() / max(calls["volume"].sum(), 1),
        "pc_oi": puts["openInterest"].sum() / max(calls["openInterest"].sum(), 1),
        "gex_musd_1pct": gex / 1e6,
    }

    # IV at-the-money y skew en el vencimiento más cercano a 30 días.
    dias = (pd.to_datetime(c["vencimiento"]) - hoy).dt.days
    if len(dias):
        objetivo = dias.iloc[(dias - 30).abs().argsort().iloc[0]]
        v = c[dias == objetivo]
        vc, vp = v[v["tipo"] == "call"], v[v["tipo"] == "put"]
        if len(vc) and len(vp):
            atm = vc.iloc[(vc["strike"] - spot).abs().argsort().iloc[0]]["impliedVolatility"]
            otm = vp.iloc[(vp["strike"] - 0.95 * spot).abs().argsort().iloc[0]]["impliedVolatility"]
            salida.update({"dias_venc_30": float(objetivo), "iv_atm_30": float(atm),
                           "skew_95_30": float(otm - atm)})
    return salida


def snapshot(ticker: str, max_dias: int = 60, max_vencimientos: int = 10) -> dict[str, float]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"{ticker}: sin precio")
    spot = float(hist["Close"].iloc[-1])
    hoy = pd.Timestamp(datetime.now().date())
    partes = []
    for venc in list(tk.options)[:max_vencimientos]:
        if (pd.Timestamp(venc) - hoy).days > max_dias:
            break
        ch = tk.option_chain(venc)
        for tipo, df in (("call", ch.calls), ("put", ch.puts)):
            d = df[["strike", "volume", "openInterest", "impliedVolatility"]].copy()
            d["tipo"], d["vencimiento"] = tipo, venc
            partes.append(d)
    if not partes:
        raise RuntimeError(f"{ticker}: sin cadenas de opciones")
    return resumir_cadena(pd.concat(partes, ignore_index=True), spot, hoy)


def snapshot_varios(tickers: list[str]) -> dict[str, float]:
    """Snapshot plano {TICKER_metrica: valor} para el registro."""
    salida: dict[str, float] = {}
    for t in tickers:
        try:
            for k, v in snapshot(t).items():
                salida[f"{t}_{k}"] = v
        except Exception as e:  # noqa: BLE001
            log.warning("Opciones %s: %s", t, e)
    return salida


def snapshot_etfs(tickers: list[str]) -> dict[str, float]:
    """Activos y acciones en circulación de ETFs (para estimar flujos a futuro)."""
    import yfinance as yf

    salida: dict[str, float] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            for campo, nombre in (("totalAssets", "activos"), ("sharesOutstanding", "acciones")):
                if info.get(campo):
                    salida[f"{t}_{nombre}"] = float(info[campo])
        except Exception as e:  # noqa: BLE001
            log.warning("ETF %s: %s", t, e)
    return salida
