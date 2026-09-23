"""Acceso de alto nivel a series ya descargadas, con las reglas point-in-time aplicadas."""

from __future__ import annotations

import pandas as pd

from ..config import Proyecto
from .almacen import Almacen


class Series:
    """Punto de acceso único a los datos para indicadores y análisis."""

    def __init__(self, proyecto: Proyecto, almacen: Almacen | None = None):
        self.p = proyecto
        self.alm = almacen or Almacen(proyecto.dir_datos)
        self._cache: dict[str, pd.DataFrame] = {}

    def _tabla(self, nombre: str) -> pd.DataFrame:
        if nombre not in self._cache:
            self._cache[nombre] = self.alm.leer(nombre)
        return self._cache[nombre]

    # --- precios ---------------------------------------------------------
    @property
    def cierres(self) -> pd.DataFrame:
        return self._tabla("precios/cierre_aj")

    @property
    def calendario(self) -> pd.DatetimeIndex:
        """Ruedas de EE. UU. (fechas con cierre de SPY)."""
        return pd.DatetimeIndex(self.cierres["SPY"].dropna().index)

    def precio(self, ticker: str) -> pd.Series:
        c = self.cierres
        return c[ticker].dropna() if ticker in c.columns else pd.Series(dtype=float, name=ticker)

    def en_calendario(self, tickers: list[str] | None = None, limite_ffill: int = 5) -> pd.DataFrame:
        """Cierres alineados a las ruedas de EE. UU. (último dato conocido, hasta `limite_ffill` ruedas)."""
        c = self.cierres if tickers is None else self.cierres.reindex(columns=tickers)
        cal = self.calendario
        union = c.index.union(cal)
        return c.reindex(union).ffill(limit=limite_ffill).reindex(cal)

    # --- volatilidad (CBOE primario, Yahoo respaldo) ---------------------
    def vol(self, sym: str) -> pd.Series:
        """Serie de la familia VIX: CBOE, completada con Yahoo (^SYM) donde falte."""
        cboe = self._tabla("cboe")
        base = cboe[sym].dropna() if sym in cboe.columns else pd.Series(dtype=float)
        yahoo = self.precio(f"^{sym}")
        if base.empty:
            return yahoo.rename(sym)
        return base.combine_first(yahoo).rename(sym)

    # --- FRED con demora de publicación ----------------------------------
    def fred(self, serie: str, respetar_lag: bool = True) -> pd.Series:
        """Serie FRED en ruedas de EE. UU. Con `respetar_lag`, cada dato aparece recién cuando
        se publicó (fecha del dato + lag_dias)."""
        tabla = self._tabla("fred")
        if serie not in tabla.columns:
            return pd.Series(dtype=float, name=serie)
        s = tabla[serie].dropna()
        if respetar_lag:
            lag = int(self.p.series_fred.get(serie, {}).get("lag_dias", 1))
            s.index = s.index + pd.Timedelta(days=lag)
        cal = self.calendario
        # Relleno limitado: una serie discontinuada no se arrastra indefinidamente.
        return s.reindex(s.index.union(cal)).ffill(limit=70).reindex(cal).rename(serie)

    # --- Ken French --------------------------------------------------------
    def french(self, nombre: str) -> pd.DataFrame:
        return self._tabla(f"french/{nombre}")
