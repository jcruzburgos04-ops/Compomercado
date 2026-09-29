"""Posiciones en futuros por tipo de operador: informe Traders in Financial Futures (TFF) de la CFTC.

Cada martes la CFTC toma la foto de las posiciones abiertas y la publica el viernes. El TFF separa a
los operadores en cuatro grupos (más los chicos que no reportan):

- **Dealers / intermediarios**: bancos y agentes que hacen de contraparte (venden cobertura,
  arman productos).
- **Asset managers**: fondos de pensión, aseguradoras, fondos comunes y ETFs (dinero institucional
  real).
- **Leveraged funds**: fondos de cobertura, CTAs y trading apalancado.
- **Otros reportables**: tesorerías de empresas, bancos centrales, family offices grandes.

Fuente principal: la API pública de la CFTC (Socrata, sin clave). Respaldo: los archivos anuales
comprimidos de cftc.gov. Historia desde junio de 2006. Nunca se completa un dato que falta.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

URL_API = "https://publicreporting.cftc.gov/resource/gpe5-46if.csv"   # TFF, solo futuros
URL_ZIP_HIST = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2006_2016.zip"
URL_ZIP_ANIO = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{anio}.zip"
CABECERAS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) compomercado/0.1", "Accept": "text/csv"}
PRIMER_ANIO = 2006

# Grupos del TFF: prefijo de columna en el informe -> clave corta.
CATEGORIAS = {"dealer": "dealer", "asset_mgr": "am", "lev_money": "lev", "other_rept": "otros", "nonrept": "nr"}
NOMBRES_CATEGORIAS = {"dealer": "Dealers (bancos)", "am": "Asset managers", "lev": "Fondos apalancados",
                      "otros": "Otros reportables", "nr": "No reportables (chicos)"}


# Mercados que se siguen. Cada uno tiene códigos de contrato conocidos y un patrón del nombre por si la
# CFTC cambia el código (pasó, por ejemplo, con el Russell al mudarse de ICE a CME). Si en una misma
# semana hay más de un contrato del mercado, se usa el de mayor interés abierto (el consolidado incluye
# al mini).
MERCADOS: dict[str, dict] = {
    "sp500": {"nombre": "S&P 500", "codigos": ["13874+", "13874A"],
              "patron": r"^(E-MINI S&P 500( STOCK INDEX)?|S&P 500 CONSOLIDATED) -"},
    "nasdaq": {"nombre": "Nasdaq 100", "codigos": ["20974+", "209742"],
               "patron": r"^(NASDAQ-100 STOCK INDEX \(MINI\)|NASDAQ MINI|NASDAQ-100 CONSOLIDATED|E-MINI NASDAQ-100) -"},
    "russell": {"nombre": "Russell 2000", "codigos": ["239742", "23977A"],
                "patron": r"^(RUSSELL 2000 MINI( INDEX FUTURE)?|RUSSELL E-MINI|E-MINI RUSSELL 2000( INDEX)?) -"},
    "vix": {"nombre": "VIX", "codigos": ["1170E1"], "patron": r"^(VIX FUTURES|CBOE VIX FUTURES) -"},
    "tesoro10": {"nombre": "Tesoro 10 años", "codigos": ["043602"],
                 "patron": r"^(10-YEAR U\.S\. TREASURY NOTES|UST 10Y NOTE) -"},
    "tesoro2": {"nombre": "Tesoro 2 años", "codigos": ["042601"],
                "patron": r"^(2-YEAR U\.S\. TREASURY NOTES|UST 2Y NOTE) -"},
    "tesoro30": {"nombre": "Tesoro 30 años", "codigos": ["020601"],
                 "patron": r"^(U\.S\. TREASURY BONDS|UST BOND) -"},
    "yen": {"nombre": "Yen", "codigos": ["097741"], "patron": r"^JAPANESE YEN -"},
    "euro": {"nombre": "Euro", "codigos": ["099741"], "patron": r"^EURO FX -"},
    "dolar": {"nombre": "Índice dólar", "codigos": ["098662"], "patron": r"^(U\.S\. DOLLAR INDEX|USD INDEX) -"},
    "bitcoin": {"nombre": "Bitcoin", "codigos": ["133741"], "patron": r"^BITCOIN - CHICAGO MERCANTILE"},
}


def _normalizar(nombre: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(nombre).strip().lower()).strip("_")


def _buscar(columnas: list[str], patron: str) -> str | None:
    """Primera columna (ya normalizada) que cumple el patrón completo."""
    rx = re.compile(patron)
    return next((c for c in columnas if rx.fullmatch(c)), None)


def normalizar(crudo: pd.DataFrame) -> pd.DataFrame:
    """Informe TFF (formato de la API o de los archivos anuales) a una tabla larga uniforme:
    fecha (martes del informe), codigo, mercado_cftc, oi y largos/cortos por grupo."""
    df = crudo.copy()
    df.columns = [_normalizar(c) for c in df.columns]
    cols = list(df.columns)
    c_fecha = _buscar(cols, r"report_date_as_yyyy_mm_dd") or _buscar(cols, r"report_date.*")
    c_codigo = _buscar(cols, r"cftc_contract_market_code")
    c_nombre = _buscar(cols, r"market_and_exchange_names?")
    c_oi = _buscar(cols, r"open_interest_all") or _buscar(cols, r"open_interest")
    faltan = [n for n, c in (("fecha", c_fecha), ("código", c_codigo), ("nombre", c_nombre), ("interés abierto", c_oi))
              if c is None]
    if faltan:
        raise ValueError(f"TFF: faltan columnas {faltan}; hay {cols[:25]}")
    salida = pd.DataFrame({
        "fecha": pd.to_datetime(df[c_fecha].astype(str).str[:10], errors="coerce"),
        "codigo": df[c_codigo].astype(str).str.strip().str.upper(),
        "mercado_cftc": df[c_nombre].astype(str).str.strip().str.upper(),
        "oi": pd.to_numeric(df[c_oi], errors="coerce"),
    })
    for prefijo, clave in CATEGORIAS.items():
        for lado, corto in (("long", "l"), ("short", "s")):
            c = _buscar(cols, rf"{prefijo}_positions_{lado}(_all)?")
            if c is None:
                raise ValueError(f"TFF: falta la columna de {prefijo} {lado}")
            salida[f"{clave}_{corto}"] = pd.to_numeric(df[c], errors="coerce")
    return salida.dropna(subset=["fecha", "oi"]).reset_index(drop=True)


def resolver(larga: pd.DataFrame, mercados: dict[str, dict] = MERCADOS) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Tabla ancha semanal (fecha x "mercado__campo") y, por mercado, los contratos usados."""
    partes, usados = {}, {}
    for clave, m in mercados.items():
        por_codigo = larga["codigo"].isin(m["codigos"])
        rx = re.compile(m["patron"])
        por_nombre = larga["mercado_cftc"].map(lambda n, rx=rx: bool(rx.search(str(n))))
        sub = larga[por_codigo | por_nombre]
        if sub.empty:
            continue
        usados[clave] = sorted(f"{c} {n}" for c, n in sub[["codigo", "mercado_cftc"]].drop_duplicates().itertuples(index=False))
        # Por semana, el contrato con más interés abierto.
        sub = sub.sort_values(["fecha", "oi"]).drop_duplicates("fecha", keep="last").set_index("fecha")
        campos = [c for c in sub.columns if c not in ("codigo", "mercado_cftc")]
        partes[clave] = sub[campos]
    if not partes:
        return pd.DataFrame(), usados
    ancha = pd.concat(partes, axis=1)
    ancha.columns = [f"{m}__{c}" for m, c in ancha.columns]
    return ancha.sort_index(), usados


# --- descarga ------------------------------------------------------------------------------------

def _where(mercados: dict[str, dict]) -> str:
    codigos = sorted({c for m in mercados.values() for c in m["codigos"]})
    lista = ",".join(f"'{c}'" for c in codigos)
    nombres = ["E-MINI S&P%", "S&P 500 CONSOLIDATED%", "NASDAQ%", "E-MINI NASDAQ%", "RUSSELL%", "E-MINI RUSSELL%",
               "VIX%", "CBOE VIX%", "10-YEAR U.S. TREASURY NOTES%", "UST 10Y NOTE%", "2-YEAR U.S. TREASURY NOTES%",
               "UST 2Y NOTE%", "U.S. TREASURY BONDS%", "UST BOND%", "JAPANESE YEN%", "EURO FX -%", "U.S. DOLLAR INDEX%",
               "USD INDEX%", "BITCOIN - CHICAGO%"]
    por_nombre = " OR ".join(f"upper(market_and_exchange_names) like '{n}'" for n in nombres)
    return f"cftc_contract_market_code in({lista}) OR {por_nombre}"


def descargar_api(timeout: int = 180, pagina: int = 50000) -> pd.DataFrame:
    """Todas las semanas de los mercados seguidos desde la API de la CFTC (tabla larga normalizada)."""
    partes, desde = [], 0
    while True:
        params = {"$where": _where(MERCADOS), "$order": "report_date_as_yyyy_mm_dd,cftc_contract_market_code",
                  "$limit": pagina, "$offset": desde}
        r = requests.get(URL_API, params=params, headers=CABECERAS, timeout=timeout)
        r.raise_for_status()
        crudo = pd.read_csv(io.StringIO(r.text), dtype=str)
        if crudo.empty:
            break
        partes.append(normalizar(crudo))
        if len(crudo) < pagina:
            break
        desde += pagina
    if not partes:
        raise ValueError("TFF: la API no devolvió filas")
    return pd.concat(partes, ignore_index=True)


def _leer_zip(contenido: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        nombres = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))]
        if not nombres:
            raise ValueError("TFF: el zip no tiene datos")
        return pd.concat([pd.read_csv(z.open(n), dtype=str, low_memory=False) for n in nombres], ignore_index=True)


def descargar_archivos(timeout: int = 180) -> pd.DataFrame:
    """Respaldo: archivos anuales de cftc.gov (2006-2016 en uno solo, después uno por año)."""
    urls = [URL_ZIP_HIST] + [URL_ZIP_ANIO.format(anio=a) for a in range(2017, date.today().year + 1)]
    partes = []
    for url in urls:
        try:
            r = requests.get(url, headers=CABECERAS, timeout=timeout)
            r.raise_for_status()
            partes.append(normalizar(_leer_zip(r.content)))
        except Exception as e:  # noqa: BLE001 - un año que falta no invalida los demás
            log.warning("TFF %s: %s", url.rsplit("/", 1)[-1], e)
    if not partes:
        raise ValueError("TFF: no se pudo leer ningún archivo anual")
    return pd.concat(partes, ignore_index=True).drop_duplicates(["fecha", "codigo"], keep="last")


def descargar() -> tuple[pd.DataFrame, list[str], str]:
    """(tabla ancha semanal, mercados sin datos, fuente usada)."""
    fuente = "API pública de la CFTC (publicreporting.cftc.gov)"
    try:
        larga = descargar_api()
    except Exception as e:  # noqa: BLE001
        log.warning("TFF API falló (%s); se prueban los archivos anuales", e)
        larga = descargar_archivos()
        fuente = "Archivos anuales de cftc.gov"
    ancha, usados = resolver(larga)
    for clave, contratos in usados.items():
        log.info("TFF %s: %s", clave, "; ".join(contratos))
    faltan = [m for m in MERCADOS if m not in usados]
    if not ancha.empty:
        log.info("TFF: %d semanas (%s a %s), %d mercados", len(ancha), ancha.index.min().date(),
                 ancha.index.max().date(), len(usados))
    return ancha, faltan, fuente


# --- métricas ------------------------------------------------------------------------------------

def mercado(ancha: pd.DataFrame, clave: str) -> pd.DataFrame:
    """Columnas de un mercado sin el prefijo (vacío si no está)."""
    cols = [c for c in ancha.columns if c.startswith(f"{clave}__")]
    t = ancha[cols].copy()
    t.columns = [c.split("__", 1)[1] for c in cols]
    return t.dropna(subset=["oi"]) if "oi" in t else t


def neto(t: pd.DataFrame, grupo: str) -> pd.Series:
    """Posición neta (largos − cortos) del grupo como fracción del interés abierto."""
    if t.empty or f"{grupo}_l" not in t:
        return pd.Series(dtype=float)
    return ((t[f"{grupo}_l"] - t[f"{grupo}_s"]) / t["oi"].where(t["oi"] > 0)).rename(grupo)


def z_movil(s: pd.Series, semanas: int = 156, minimo: int = 52) -> pd.Series:
    """Desvíos del valor contra la media de las `semanas` anteriores (incluida la actual)."""
    m = s.rolling(semanas, min_periods=minimo).mean()
    sd = s.rolling(semanas, min_periods=minimo).std()
    return (s - m) / sd.where(sd > 0)


def indice_cot(s: pd.Series, semanas: int = 156, minimo: int = 52) -> pd.Series:
    """Índice COT (0 = el neto más bajo de 3 años, 1 = el más alto)."""
    lo = s.rolling(semanas, min_periods=minimo).min()
    hi = s.rolling(semanas, min_periods=minimo).max()
    return ((s - lo) / (hi - lo).where(hi > lo)).clip(0, 1)


def lectura(z: float) -> str:
    if z is None or np.isnan(z):
        return ""
    if z >= 2:
        return "extremo comprado"
    if z >= 1:
        return "más comprado que lo habitual"
    if z <= -2:
        return "extremo vendido"
    if z <= -1:
        return "más vendido que lo habitual"
    return "normal"
