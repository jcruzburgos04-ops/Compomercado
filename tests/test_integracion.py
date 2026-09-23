import numpy as np
import pandas as pd

from compomercado import registro
from compomercado.datos.almacen import Almacen
from compomercado.datos.series import Series
from compomercado.indicadores import estado
from compomercado.pipeline import analizar
from compomercado.reportes.dashboard import construir

from . import sintetico


def test_pipeline_y_dashboard_de_punta_a_punta(tmp_path):
    p = sintetico.crear(tmp_path)
    res = analizar(p, registrar=True, snapshot_opciones=False)
    assert "SPY" in res.mapas and len(res.mapas["SPY"].episodios) >= 1
    assert res.pilares["total"].dropna().between(0, 100).all()
    assert res.historia_larga is not None and not res.historia_larga.tabla.empty
    assert any(c.startswith("canasta:") for c in res.canastas)
    assert res.huellas is not None and len(res.huellas.tramos) >= 1
    assert res.huellas.prob.dropna().between(0, 1).all()

    ruta = construir(res, p.dir_sitio)
    html = ruta.read_text(encoding="utf-8")
    for seccion in ("Estado del mercado", "Mapa de comportamiento", "Huellas y análogos", "Historia desde 1926",
                    "Registro forward"):
        assert seccion in html
    assert "Argentina" not in html and "CCL" not in html
    assert "Esta sección falló" not in html
    # Los datos del test son sintéticos: el tablero tiene que avisarlo.
    assert "DATOS DE PRUEBA" in html and "Datos reales descargados" not in html
    assert (p.dir_sitio / "datos" / "mapa_SPY.csv").exists()

    # El registro forward no duplica la fila del día.
    analizar(p, registrar=True, snapshot_opciones=False)
    assert len(registro.leer(p.dir_registro, "estado_diario.csv")) == 1


def test_indicadores_sin_look_ahead(tmp_path):
    """Recalcular con datos truncados a una fecha pasada da el mismo valor en esa fecha."""
    p = sintetico.crear(tmp_path / "completo")
    completos = {i.id: i.serie for i in estado.calcular(Series(p))}

    corte = pd.Timestamp("2021-06-15")
    q = sintetico.crear(tmp_path / "truncado")
    alm = Almacen(q.dir_datos)
    for nombre in ["precios/cierre_aj", "precios/cierre", "precios/apertura", "precios/maximo", "precios/minimo",
                   "precios/volumen", "cboe"]:
        alm.guardar(nombre, alm.leer(nombre).loc[:corte])
    # FRED: el dato con fecha d se publica en d + lag, así que a la fecha de corte solo se conocía hasta corte - lag.
    fred = alm.leer("fred")
    alm.guardar("fred", fred.loc[:corte])
    truncados = {i.id: i.serie for i in estado.calcular(Series(q))}

    fecha = truncados["vix"].index.max()
    assert fecha <= corte
    for id_, s in truncados.items():
        a, b = completos[id_].get(fecha, np.nan), s.get(fecha, np.nan)
        if np.isnan(a) and np.isnan(b):
            continue
        assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{id_}: {a} != {b}"

    # Los percentiles de riesgo tampoco miran el futuro.
    r_full, _ = estado.puntajes(estado.calcular(Series(p)))
    r_trunc, _ = estado.puntajes(estado.calcular(Series(q)))
    comunes = r_trunc.columns.intersection(r_full.columns)
    assert np.allclose(r_full.loc[fecha, comunes].astype(float), r_trunc.loc[fecha, comunes].astype(float), equal_nan=True)


def test_parte_diario_en_texto(tmp_path):
    from compomercado.reportes.resumen import texto

    p = sintetico.crear(tmp_path)
    res = analizar(p, registrar=False, snapshot_opciones=False)
    t = texto(res)
    assert "RIESGO TOTAL" in t and "MAPA VS SPY" in t and "HISTORIA DESDE 1926" in t
    assert "ARGENTINA" not in t
    assert "HUELLAS DE LAS CAÍDAS" in t


def test_sello_de_origen_real(tmp_path):
    from compomercado.datos.almacen import Almacen

    alm = Almacen(tmp_path)
    assert not alm.origen()["real"]
    for fuente in ("yahoo", "fred", "cboe", "ken_french"):
        alm.registrar_descarga(fuente, origen="descarga", fallidos=[])
    assert alm.origen()["real"]
    alm.registrar_descarga("fred", origen="sintetico", fallidos=[])
    assert not alm.origen()["real"]
