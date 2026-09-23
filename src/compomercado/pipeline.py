"""Corre el análisis completo sobre los datos descargados y devuelve todo lo que usa el dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import registro
from .analitica import canastas, huellas
from .analitica import comportamiento as comp
from .config import Proyecto
from .datos.series import Series
from .eventos import caidas
from .indicadores import estado, ponderacion

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
                 "XLB", "XLRE", "XLC", "SMH", "KRE", "TLT", "HYG", "LQD", "GLD"]


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
    pilares: pd.DataFrame            # principal: ponderados por importancia (fuera de muestra) si hay historia
    pilares_igual: pd.DataFrame      # pesos iguales (esquema anterior, para comparar)
    ponderacion: ponderacion.Ponderacion | None
    tabla_base: pd.DataFrame
    mapas: dict[str, MapaRef]
    bajo_agua: pd.DataFrame
    historia_larga: MapaRef | None
    correlacion: pd.DataFrame
    canastas: list[str]
    calidad: pd.DataFrame
    metadatos: dict
    origen: dict
    registro_estado: pd.DataFrame
    registro_opciones: pd.DataFrame
    nombres: dict[str, str]
    ultimas_fechas: dict = field(default_factory=dict)
    umbral_tramos: float = 0.05
    huellas: huellas.Huellas | None = None


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
    riesgo, pilares_igual = estado.puntajes(indicadores)
    spy = precios["SPY"]
    baselines = {"SPY bajo su media de 200": -(spy / spy.rolling(200).mean() - 1),
                 "VIX": S.vol("VIX").reindex(cal)}
    pond = ponderacion.analizar(indicadores, riesgo, pilares_igual.get("total", pd.Series(dtype=float)),
                                spy, baselines)
    pilares = pond.pilares if pond is not None else pilares_igual
    tabla_base = estado.tabla_base(pilares["total"], spy) if "total" in pilares else pd.DataFrame()

    # --- huellas de las caídas y análogos ---------------------------------------------------------
    huellas_spy = None
    if "SPY" in mapas and "total" in pilares:
        huellas_spy = huellas.analizar(mapas["SPY"].episodios, riesgo, pilares_igual, pilares["total"], spy)

    # --- correlaciones actuales ---------------------------------------------------------------
    corr_cols = [t for t in CORRELACION if t in precios.columns]
    correlacion = precios[corr_cols].pct_change().tail(63).corr()


    # --- registro forward -----------------------------------------------------------------------
    ahora_ny = datetime.now(ZoneInfo("America/New_York"))
    hoy_ny = pd.Timestamp(ahora_ny.date())
    rueda_cerrada = ahora_ny.weekday() < 5 and (ahora_ny.hour, ahora_ny.minute) >= (16, 30)
    if registrar:
        valores = {f"ind_{i.id}": _ultimo(i.serie) for i in indicadores}
        # pilar_* = pesos iguales (así se registró desde el primer día); pond_* = ponderado.
        valores.update({f"pilar_{k}": _ultimo(pilares_igual[k]) for k in pilares_igual.columns})
        if pond is not None:
            valores.update({f"pond_{k}": _ultimo(pond.pilares[k]) for k in pond.pilares.columns})
        if huellas_spy is not None:
            valores["analogos_prob"] = huellas_spy.vecinos_hoy.get("prob", np.nan)
        valores["spy_cierre"] = _ultimo(precios["SPY"])
        if registro.agregar(proyecto.dir_registro, "estado_diario.csv", fecha, valores):
            log.info("Registro forward: fila %s agregada", fecha.date())
        # Un snapshot refleja la última rueda cerrada: la de hoy si ya pasó el cierre, o la
        # anterior si la corrida es antes de la apertura (la corrida diaria es a la mañana).
        # Durante la rueda las cadenas son intradía: no se registran.
        fecha_snap = hoy_ny if rueda_cerrada else fecha
        en_rueda = ahora_ny.weekday() < 5 and (9, 30) <= (ahora_ny.hour, ahora_ny.minute) < (16, 30)
        if snapshot_opciones and not en_rueda:
            from .datos import opciones

            snap = opciones.snapshot_varios(["SPY", "QQQ", "IWM"])
            if snap:
                registro.agregar(proyecto.dir_registro, "opciones_diario.csv", fecha_snap, snap)
            etfs = opciones.snapshot_etfs(ETFS_SNAPSHOT)
            if etfs:
                registro.agregar(proyecto.dir_registro, "etfs_diario.csv", fecha_snap, etfs)

    calidad_path = proyecto.dir_datos / "calidad_precios.csv"
    calidad = pd.read_csv(calidad_path, index_col=0) if calidad_path.exists() else pd.DataFrame()

    return Resultados(
        fecha=fecha,
        precios=precios,
        indicadores=indicadores,
        riesgo=riesgo,
        pilares=pilares,
        pilares_igual=pilares_igual,
        ponderacion=pond,
        tabla_base=tabla_base,
        mapas=mapas,
        bajo_agua=bajo_agua,
        historia_larga=historia_larga,
        correlacion=correlacion,
        canastas=nombres_canastas,
        calidad=calidad,
        metadatos=S.alm.metadatos(),
        origen=S.alm.origen(),
        registro_estado=registro.leer(proyecto.dir_registro, "estado_diario.csv"),
        registro_opciones=_opciones_validas(registro.leer(proyecto.dir_registro, "opciones_diario.csv")),
        nombres=nombres,
        ultimas_fechas=_ultimas_fechas(S),
        umbral_tramos=umbral,
        huellas=huellas_spy,
    )


def _ultimas_fechas(S: Series) -> dict:
    """Último dato disponible de cada fuente, para detectar demoras (p. ej. Yahoo sin la última rueda)."""
    salida = {}
    for etiqueta, serie in (("SPY (Yahoo)", S.precio("SPY")), ("VIX (CBOE)", S._tabla("cboe").get("VIX")),
                            ("Nikkei (Yahoo)", S.precio("^N225")),
                            ("Tasa 10a (FRED)", S._tabla("fred").get("DGS10"))):
        if serie is not None and serie.notna().any():
            salida[etiqueta] = serie.last_valid_index()
    return salida


def _opciones_validas(o: pd.DataFrame) -> pd.DataFrame:
    """Descarta las métricas de un ticker en los snapshots sin precio spot válido.

    El registro no se reescribe: los snapshots inválidos quedan guardados y se ignoran al leer.
    """
    if o.empty:
        return o
    o = o.copy()
    for t in {c.split("_")[0] for c in o.columns if c.endswith("_spot")}:
        invalido = pd.to_numeric(o[f"{t}_spot"], errors="coerce").isna()
        cols = [c for c in o.columns if c.startswith(f"{t}_")]
        o.loc[invalido, cols] = np.nan
    return o


def _ultimo(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan
