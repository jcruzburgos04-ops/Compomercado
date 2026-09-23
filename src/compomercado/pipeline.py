"""Corre el análisis completo sobre los datos descargados y devuelve todo lo que usa el dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import registro
from .analitica import argentina, canastas
from .analitica import comportamiento as comp
from .config import Proyecto
from .datos.series import Series
from .eventos import caidas
from .indicadores import estado

log = logging.getLogger(__name__)

INDUSTRIAS_FF = {
    "Agric": "Agricultura", "Food": "Alimentos", "Soda": "Golosinas y gaseosas", "Beer": "Bebidas alcohólicas",
    "Smoke": "Tabaco", "Toys": "Recreación", "Fun": "Entretenimiento", "Books": "Imprenta y editoriales",
    "Hshld": "Bienes de consumo", "Clths": "Indumentaria", "Hlth": "Servicios de salud", "MedEq": "Equipos médicos",
    "Drugs": "Farmacéuticas", "Chems": "Químicas", "Rubbr": "Caucho y plásticos", "Txtls": "Textiles",
    "BldMt": "Materiales de construcción", "Cnstr": "Construcción", "Steel": "Acero", "FabPr": "Productos fabricados",
    "Mach": "Maquinaria", "ElcEq": "Equipos eléctricos", "Autos": "Automotrices", "Aero": "Aeronáutica",
    "Ships": "Naval y ferroviario", "Guns": "Defensa", "Gold": "Metales preciosos", "Mines": "Minería",
    "Coal": "Carbón", "Oil": "Petróleo y gas", "Util": "Utilities", "Telcm": "Comunicaciones",
    "PerSv": "Servicios personales", "BusSv": "Servicios empresariales", "Hardw": "Computadoras",
    "Softw": "Software", "Chips": "Electrónica / chips", "LabEq": "Instrumental", "Paper": "Insumos de oficina",
    "Boxes": "Envases", "Trans": "Transporte", "Whlsl": "Mayoristas", "Rtail": "Minoristas",
    "Meals": "Restaurantes y hoteles", "Banks": "Bancos", "Insur": "Seguros", "RlEst": "Inmobiliario",
    "Fin": "Servicios financieros", "Other": "Otros",
}

CORRELACION = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU",
               "XLB", "XLRE", "XLC", "SMH", "KRE", "TLT", "IEF", "HYG", "LQD", "GLD", "UUP", "DBC"]

ETFS_SNAPSHOT = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU",
                 "XLB", "XLRE", "XLC", "SMH", "KRE", "TLT", "HYG", "LQD", "GLD", "ARGT"]


@dataclass
class MapaRef:
    ref: str
    episodios: pd.DataFrame
    tabla: pd.DataFrame
    larga: pd.DataFrame
    por_tipo: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class Resultados:
    fecha: pd.Timestamp
    precios: pd.DataFrame
    indicadores: list
    riesgo: pd.DataFrame
    pilares: pd.DataFrame
    tabla_base: pd.DataFrame
    mapas: dict[str, MapaRef]
    bajo_agua: pd.DataFrame
    historia_larga: MapaRef | None
    correlacion: pd.DataFrame
    argentina: dict
    canastas: list[str]
    calidad: pd.DataFrame
    metadatos: dict
    registro_estado: pd.DataFrame
    registro_opciones: pd.DataFrame
    nombres: dict[str, str]


def _nombre_canasta(nombre: str) -> str:
    return f"canasta:{nombre}"


def analizar(proyecto: Proyecto, registrar: bool = True, snapshot_opciones: bool = True) -> Resultados:
    S = Series(proyecto)
    cfg = proyecto.analisis
    cal = S.calendario
    fecha = cal.max()
    log.info("Datos hasta %s", fecha.date())

    # --- universo alineado al calendario de EE. UU. + canastas ---------------------------------
    grupos = cfg.get("grupos_mapa", [])
    universo = [i.ticker for i in proyecto.instrumentos if i.grupo in grupos]
    precios = S.en_calendario()
    nombres = dict(proyecto.nombres)
    nombres_canastas = []
    for c in proyecto.canastas:
        serie = canastas.serie_canasta(precios, c)
        if serie.dropna().empty:
            continue
        col = _nombre_canasta(c.nombre)
        precios[col] = serie.reindex(cal)
        nombres[col] = c.descripcion
        nombres_canastas.append(col)
    universo_mapa = [t for t in universo + nombres_canastas if t in precios.columns]
    asinc = {t for t in universo_mapa if proyecto.es_asincronico(t)}

    # --- episodios de SPY con huella macro --------------------------------------------------
    cf = cfg.get("caidas", {})
    umbral = float(cf.get("umbral_mapa", 0.10))
    pre = int(cf.get("ventana_pre_pico", 63))
    rebotes = tuple(cf.get("ventana_rebote", [21, 63]))
    huella_niveles = {"dgs10": S.fred("DGS10", respetar_lag=False), "baa10y": S.fred("BAA10Y", respetar_lag=False)}
    huella_precios = {"dxy": precios.get("DX-Y.NYB"), "wti": precios.get("CL=F"), "usdjpy": precios.get("JPY=X"),
                      "oro": precios.get("GC=F")}
    huella_precios = {k: v for k, v in huella_precios.items() if v is not None}
    huella_max = {"vix": S.vol("VIX").reindex(cal)}

    mapas: dict[str, MapaRef] = {}
    for ref in cfg.get("referencias_mapa", ["SPY"]):
        if ref not in precios.columns:
            continue
        pm = precios[ref].dropna()
        eps = caidas.tramos_zigzag(pm, umbral)
        if ref == "SPY":
            eps = eps.join(caidas.huella_macro(eps, huella_niveles, huella_precios, huella_max))
            eps["tipo"] = caidas.clasificar(eps, cfg.get("tipos_caida", []))
        log.info("Mapa vs %s: %d episodios", ref, len(eps))
        tabla, larga = comp.mapa(precios[universo_mapa].loc[pm.index[0]:], pm, eps, asinc, pre=pre, rebotes=rebotes)
        tabla["grupo"] = [proyecto.grupo_de.get(t, "canasta" if t.startswith("canasta:") else "") for t in tabla.index]
        mapa = MapaRef(ref, eps, tabla, larga)
        if "tipo" in eps:
            mapa.por_tipo = comp.por_tipo(larga, eps["tipo"])
        mapas[ref] = mapa

    bajo_agua = caidas.episodios_bajo_agua(precios["SPY"].dropna(), 0.10)

    # --- historia larga: 49 industrias Fama-French desde 1926 -----------------------------------
    historia_larga = None
    ind = S.french("49_Industry_Portfolios_daily")
    fac = S.french("F-F_Research_Data_Factors_daily")
    if not ind.empty and not fac.empty:
        pm_ff = (1 + fac["Mkt-RF"] + fac["RF"]).cumprod().rename("Mercado")
        precios_ff = (1 + ind).cumprod().reindex(pm_ff.index)
        precios_ff.columns = [INDUSTRIAS_FF.get(c, c) for c in precios_ff.columns]
        eps_ff = caidas.episodios_bajo_agua(pm_ff, 0.20)
        tabla_ff, larga_ff = comp.mapa(precios_ff, pm_ff, eps_ff, pre=pre, rebotes=rebotes)
        historia_larga = MapaRef("Mercado EE. UU. (Fama-French)", eps_ff, tabla_ff, larga_ff)
        log.info("Historia larga: %d episodios desde %s", len(eps_ff), pm_ff.index[0].date())

    # --- panel de estado ----------------------------------------------------------------------
    indicadores = estado.calcular(S)
    riesgo, pilares = estado.puntajes(indicadores)
    tabla_base = estado.tabla_base(pilares["total"], precios["SPY"]) if "total" in pilares else pd.DataFrame()

    # --- correlaciones actuales ---------------------------------------------------------------
    corr_cols = [t for t in CORRELACION if t in precios.columns]
    correlacion = precios[corr_cols].pct_change().tail(63).corr()

    # --- Argentina ----------------------------------------------------------------------------
    arg = _argentina(proyecto, S, precios, mapas.get("SPY"), asinc, pre, rebotes)

    # --- registro forward -----------------------------------------------------------------------
    hoy_ny = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    if registrar:
        valores = {f"ind_{i.id}": _ultimo(i.serie) for i in indicadores}
        valores.update({f"pilar_{k}": _ultimo(pilares[k]) for k in pilares.columns})
        valores["spy_cierre"] = _ultimo(precios["SPY"])
        if registro.agregar(proyecto.dir_registro, "estado_diario.csv", fecha, valores):
            log.info("Registro forward: fila %s agregada", fecha.date())
        if snapshot_opciones and fecha == hoy_ny:
            from .datos import opciones

            snap = opciones.snapshot_varios(["SPY", "QQQ", "IWM"])
            if snap:
                registro.agregar(proyecto.dir_registro, "opciones_diario.csv", fecha, snap)
            etfs = opciones.snapshot_etfs(ETFS_SNAPSHOT)
            if etfs:
                registro.agregar(proyecto.dir_registro, "etfs_diario.csv", fecha, etfs)

    calidad_path = proyecto.dir_datos / "calidad_precios.csv"
    calidad = pd.read_csv(calidad_path, index_col=0) if calidad_path.exists() else pd.DataFrame()

    return Resultados(
        fecha=fecha,
        precios=precios,
        indicadores=indicadores,
        riesgo=riesgo,
        pilares=pilares,
        tabla_base=tabla_base,
        mapas=mapas,
        bajo_agua=bajo_agua,
        historia_larga=historia_larga,
        correlacion=correlacion,
        argentina=arg,
        canastas=nombres_canastas,
        calidad=calidad,
        metadatos=S.alm.metadatos(),
        registro_estado=registro.leer(proyecto.dir_registro, "estado_diario.csv"),
        registro_opciones=registro.leer(proyecto.dir_registro, "opciones_diario.csv"),
        nombres=nombres,
    )


def _ultimo(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def _argentina(proyecto, S, precios, mapa_spy, asinc, pre, rebotes) -> dict:
    cfg = proyecto.argentina
    pares = cfg.get("ccl_pares", [])
    if not pares:
        return {}
    locales = precios.reindex(columns=[p["local"] for p in pares])
    adrs = precios.reindex(columns=[p["adr"] for p in pares])
    ccl, por_especie = argentina.ccl_implicito(locales, adrs, pares)
    salida: dict = {"ccl": ccl, "ccl_especies": por_especie}
    merval = precios.get(cfg.get("merval", "^MERV"))
    if merval is not None and not ccl.dropna().empty:
        salida["merval_usd"] = (merval / ccl).rename("Merval en USD (CCL)")

    col = _nombre_canasta("argentina_adrs")
    if col in precios.columns:
        r_arg = precios[col].pct_change()
        factores = precios.reindex(columns=["SPY", "EEM", "DBC"]).pct_change()
        salida["exposicion"] = argentina.exposicion_global(r_arg, factores.dropna(how="all", axis=1))

    if mapa_spy is not None:
        activos = [p["adr"] for p in pares] + [t for t in (col, "ARGT", "EWZ", "EEM") if t in precios.columns]
        activos = [t for t in activos if t in precios.columns]
        eps = mapa_spy.episodios
        pm = precios["SPY"].dropna()
        tabla_spy, _ = comp.mapa(precios[activos].loc[pm.index[0]:], pm, eps, asinc, pre=pre, rebotes=rebotes)
        salida["mapa_spy"] = tabla_spy
        if "EEM" in precios.columns:
            pe = precios["EEM"].dropna()
            eps_eem = caidas.tramos_zigzag(pe, 0.10)
            tabla_eem, _ = comp.mapa(precios[activos].loc[pe.index[0]:], pe, eps_eem, asinc, pre=pre, rebotes=rebotes)
            salida["mapa_eem"] = tabla_eem
    return salida
