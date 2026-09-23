import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fechas():
    return pd.bdate_range("2010-01-01", periods=1500)


@pytest.fixture
def mercado(fechas):
    """Precio sintético con una caída de ~25 % en tres tramos y recuperación."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.008, len(fechas))
    r[300:340] = -0.004   # tramo 1: ~ -15 %
    r[340:370] = 0.004    # rebote ~ +13 %
    r[370:420] = -0.004   # tramo 2: ~ -18 %
    r[420:520] = 0.003    # recuperación
    return pd.Series(100 * np.cumprod(1 + r), index=fechas, name="MKT")
