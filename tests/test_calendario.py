import numpy as np
import pandas as pd

from compomercado.analitica import calendario as cal


def test_vencimientos_con_feriados():
    assert cal.opex(2026, 9) == pd.Timestamp("2026-09-18")
    # Abril de 2025: el tercer viernes fue Viernes Santo, el vencimiento pasó al jueves.
    assert cal.opex(2025, 4) == pd.Timestamp("2025-04-17")
    # VIX: miércoles 30 días antes del tercer viernes del mes siguiente (oct-2026: viernes 16).
    assert cal.vencimiento_vix(2026, 9) == pd.Timestamp("2026-09-16")
    assert cal.fin_de_mes(2026, 5) == pd.Timestamp("2026-05-29")


def test_proximos_eventos_y_ruedas():
    oficiales = pd.DataFrame({"fecha": [pd.Timestamp("2026-10-28")], "evento": ["Decisión de la Fed (FOMC)"],
                              "tipo": ["fed"]})
    t = cal.proximos(pd.Timestamp("2026-09-23"), oficiales, dias=40)
    assert (t["fecha"] > pd.Timestamp("2026-09-23")).all()
    fila = t[t["tipo"] == "fed"].iloc[0]
    assert fila["ruedas"] == 25
    assert "Fin de trimestre (rebalanceo)" in set(t["evento"])


def test_estacionalidad_usa_solo_anios_previos():
    fechas = pd.bdate_range("2000-01-03", "2026-09-23")
    r = pd.Series(0.0005, index=fechas)
    r[fechas.month == 9] = -0.002
    p = (1 + r).cumprod()
    e = cal.estacionalidad(p, pd.Timestamp("2026-09-10"))
    assert e["anios"] == 26 and e["mes_positivo"] == 0.0 and e["mes_medio"] < 0
    assert np.isfinite(e["adelante_medio"])
