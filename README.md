# Compomercado

Sensor de comportamiento de mercado y sectores. Todos los días al cierre responde:

1. **¿Qué tan frágil o estresado está el mercado?** Sensor de riesgo con diez pilares: volatilidad, crédito, amplitud, tendencia, cross-asset, global, flujo institucional, sentimiento, correlación y calendario.
2. **Si el mercado cae, ¿qué cae más y qué aguanta?** Mapa de comportamiento de sectores, factores, países, acciones y canastas propias en cada caída histórica.
3. **¿Hacia dónde rotar o cómo ajustar sin frenar los trades que funcionan?** Overlay gradual, refugios con fuerza relativa, coberturas y señales de reentrada.
4. **¿Qué está haciendo el dinero grande?** COT, short volume fuera de bolsa, distribución y flujos mecánicos estimados.

Todo validado con backtesting walk-forward y sin look-ahead.

## Estado

**Fase de planificación.** Documentos:

- [ROADMAP.md](ROADMAP.md): idea, arquitectura, módulos, fases y decisiones abiertas.
- [docs/catalogo_variables.md](docs/catalogo_variables.md): variables candidatas por pilar.
- [docs/metodologia_backtest.md](docs/metodologia_backtest.md): definiciones de caída, objetivos, reglas anti-sesgo y validación.
- [config/universos.yaml](config/universos.yaml) y [config/canastas.yaml](config/canastas.yaml): universo y canastas (borrador).

Próximo paso: **Fase 0**, cimientos de datos.
