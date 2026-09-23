# Compomercado

Sensor de comportamiento de mercado y sectores. Todos los días al cierre responde:

1. **¿Qué tan frágil o estresado está el mercado?** Sensor de riesgo con diez pilares: volatilidad, crédito, amplitud, tendencia, cross-asset, global, flujo institucional, sentimiento, correlación y calendario.
2. **Si el mercado cae, ¿qué cae más y qué aguanta?** Mapa de comportamiento de sectores, factores, países, acciones y canastas propias en cada caída histórica.
3. **¿Hacia dónde rotar o cómo ajustar sin frenar los trades que funcionan?** Overlay gradual, refugios con fuerza relativa, coberturas y señales de reentrada.
4. **¿Qué está haciendo el dinero grande?** COT, short volume fuera de bolsa, distribución y flujos mecánicos estimados.

Todo validado con backtesting walk-forward y sin look-ahead.

## Dashboard

**https://jcruzburgos04-ops.github.io/Compomercado/**. Se regenera de martes a sábado a las 10:30 UTC (antes de
la apertura de EE. UU., con la rueda anterior completa) con GitHub Actions ([`diario.yml`](.github/workflows/diario.yml)).

Secciones:

- **Estado del mercado**: puntaje de riesgo por pilar (preliminar) y qué pasó históricamente después de cada nivel.
- **Mapa de comportamiento**: cada sector, industria, factor, país, activo y canasta contra SPY, QQQ, IWM, EFA y EEM. Incluye refugios con fuerza hoy y qué reducir primero.
- **Caídas**: cada tramo ≥5 % de SPY desde 1993, con su huella macro y su tipo.
- **Huellas y análogos**: cómo estaban los indicadores y pilares un mes y una semana antes, en el pico, en la confirmación y en el valle de cada caída; qué se repite; y qué días del pasado se parecen a hoy, con su probabilidad de caída evaluada fuera de muestra.
- **Historia desde 1926**: 49 industrias Fama-French en cada mercado bajista.
- **Correlaciones**: matriz actual, absorption ratio y turbulencia.
- **Argentina**: CCL implícito, Merval en dólares, cuánto explica el mundo, ADRs frente a caídas globales.
- **Registro forward**: señales guardadas día a día (rama `registro`), más snapshots propios de opciones.

## Uso local

```bash
pip install -e ".[dev]"
compomercado datos       # descarga Yahoo, FRED, CBOE y Ken French a ./datos
compomercado analizar    # corre el análisis y construye ./sitio/index.html
pytest                   # tests (incluye el test anti look-ahead)
```

Los universos, canastas propias y parámetros se editan en [`config/`](config/).

## Estado

| Fase | Estado |
|------|--------|
| F0: cimientos de datos | ✓ Yahoo, FRED (con demora de publicación), CBOE, Ken French; almacén Parquet; controles de calidad |
| F1: caídas + mapa de comportamiento | ✓ Tramos zigzag y episodios bajo el agua, huella macro, métricas por activo y episodio, puntajes refugio/rebote, historia desde 1926 |
| F2: indicadores + dashboard | En curso: ~35 indicadores en 8 pilares, dashboard en Pages, registro forward y snapshots de opciones |
| F3: sensor validado | Pendiente |
| F4: backtest de overlay y rotación | Pendiente |
| F5: flujo institucional (COT, FINRA, insiders) | Pendiente |

Documentos:

- [ROADMAP.md](ROADMAP.md): idea, arquitectura, módulos, fases y decisiones.
- [docs/catalogo_variables.md](docs/catalogo_variables.md): variables candidatas por pilar y proxies propios.
- [docs/metodologia_backtest.md](docs/metodologia_backtest.md): definiciones de caída, objetivos swing, reglas anti-sesgo, validación y forward test.
