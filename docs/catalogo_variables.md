# Catálogo de variables

Lista de **candidatas**, no de variables aprobadas. Cada una entra al sensor solo si pasa el protocolo de [`metodologia_backtest.md`](metodologia_backtest.md): aporta información fuera de muestra y su costo en alertas falsas es aceptable.

Convenciones:

- **Historia**: año aproximado de inicio de datos gratuitos. Se verifica en la Fase 0.
- **Signo**: `↑` = valores altos implican más riesgo; `↓` = valores bajos implican más riesgo; `±` = hay que testearlo o es contrarian en extremos.
- **Lag**: demora entre el dato y su publicación. Se respeta estrictamente en el backtest.
- **Fase**: en qué fase del [roadmap](../ROADMAP.md) se implementa.

---

## A. Indicadores por pilar

### Pilar 1: Volatilidad y opciones

| Variable | Qué mide / cálculo | Fuente | Historia | Signo | Fase |
|----------|--------------------|--------|----------|-------|------|
| VIX: nivel, percentil 1a/5a, variación 5d | Volatilidad implícita del S&P 500 a 30 días | CBOE / Yahoo `^VIX` | 1990 | ↑ | F2 |
| Estructura temporal: VIX9D/VIX, VIX/VIX3M, VIX3M/VIX6M | > 1 = backwardation, estrés agudo; la vuelta a < 1 es señal de reentrada | CBOE | 2011 / 2007 / 2008 | ↑ | F2 |
| VVIX | Volatilidad del VIX: demanda de coberturas de cola | CBOE | ≈2006 | ↑ | F2 |
| SKEW | Precio relativo de la cola izquierda | CBOE | 1990 | ± | F2 |
| Prima de riesgo de volatilidad (VRP) | VIX² − varianza realizada 21d (o VIX − vol realizada). Negativa = estrés real; muy baja con vol baja = complacencia | Derivado | 1990 | ± | F2 |
| Volatilidad realizada SPY (21/63d), estimadores OHLC (Parkinson, Garman-Klass), vol de vol | Riesgo realizado; se agrupa en clusters y es muy persistente | Derivado | 1993 (^GSPC antes) | ↑ | F2 |
| MOVE | Volatilidad implícita de bonos del Tesoro | Yahoo `^MOVE` | ≈2002 | ↑ | F2 |
| VXN − VIX, RVX − VIX | Estrés concentrado en tecnología o en small caps | CBOE | 2001 / 2004 | ↑ | F2 |
| OVX, GVZ | Volatilidad de petróleo y oro: shocks de commodities | CBOE | 2007 / 2008 | ↑ | F2 |
| Correlación implícita (COR1M/COR3M), dispersión (DSPX) | Correlación baja + VIX bajo = mercado concentrado y frágil | CBOE | a verificar | ± | F2 |
| Put/Call (total, acciones, índices) | Demanda de protección; contrarian en extremos | CBOE (CSV hasta 2019; luego scrape o pago) | 1995–2003 | ± | F5 |
| GEX / DIX (SqueezeMetrics) | Gamma de dealers y short volume fuera de bolsa; útil para validar nuestra versión | SqueezeMetrics (a verificar) | ≈2011 | ± | F5 |

### Pilar 2: Crédito y financiamiento

| Variable | Qué mide / cálculo | Fuente | Historia | Signo | Fase |
|----------|--------------------|--------|----------|-------|------|
| Spread Baa − Tesoro 10a | Prima de crédito corporativo | FRED `BAA10Y` | 1986 | ↑ | F2 |
| OAS High Yield e Investment Grade | Spreads ICE BofA. FRED recortó el historial de estas series: verificar cuánto queda | FRED `BAMLH0A0HYM2`, `BAMLC0A0CM` | a verificar | ↑ | F2 |
| HYG/IEF, JNK/IEF, LQD/IEF, EMB/IEF | Crédito "que no confirma" las subas de acciones | Yahoo | 2007 / 2002 | ↓ | F2 |
| Excess Bond Premium (Gilchrist-Zakrajšek) | Componente del spread no explicado por riesgo de default; muy buen predictor de recesión | Federal Reserve (mensual) | 1973 | ↑ | F2 |
| Índices de estrés financiero | St. Louis Fed (semanal), Chicago Fed NFCI/ANFCI (semanal), OFR FSI (diario) | FRED / OFR | 1993 / 1971 / 2000 | ↑ | F2 |
| Bancos regionales relativo: KRE/SPY, KBE/SPY | Estrés bancario (2023) | Yahoo | 2006 | ↓ | F2 |
| Spreads de financiamiento de corto plazo | Papel comercial financiero − T-bill, SOFR − T-bill | FRED | varias | ↑ | F2 |
| Liquidez neta de la Fed | Balance (WALCL) − TGA − repos reversos (RRP); variación 4–13 semanas | FRED | 2002 | ↓ | F2 |

### Pilar 3: Amplitud e internals

La amplitud "verdadera" necesita constituyentes históricos, incluidas las empresas deslistadas. Con datos gratuitos se calcula sobre tres universos:

- (a) ETFs de sector, industria y país: sin sesgo.
- (b) 49 industrias Fama-French: desde 1926, sin sesgo.
- (c) S&P 500 actual: **con sesgo de supervivencia**, solo para el período reciente y marcado como tal.

Norgate (pago) resuelve esto si hace falta.

| Variable | Qué mide / cálculo | Fuente | Historia | Signo | Fase |
|----------|--------------------|--------|----------|-------|------|
| % de componentes sobre 20/50/200 DMA | Participación | Derivado (a/b/c) | 1926 (b) | ↓ | F2 |
| Línea avance-descenso, McClellan Oscillator/Summation | Momentum de la amplitud | Derivado (a/c); NYSE (pago) | según universo | ↓ | F2 |
| Nuevos máximos − nuevos mínimos 52 semanas | Deterioro interno | Derivado | según universo | ↓ | F2 |
| **Divergencia de amplitud** | Índice a < X % del máximo con amplitud cayendo N ruedas | Derivado | — | ↑ | F2 |
| RSP/SPY (igual peso vs capitalización) | Concentración del avance | Yahoo | 2003 | ↓ | F2 |
| IWM/SPY, MDY/SPY | Small y mid caps vs large: apetito de riesgo | Yahoo | 2000 / 1995 | ↓ | F2 |
| Cíclicos/defensivos: XLY/XLP, XLI/XLU, SPHB/SPLV | Risk-on / risk-off interno | Yahoo | 1998 / 2011 | ↓ | F2 |
| Sectores canario: SMH/SPY, IYT/SPY (teoría de Dow), KRE/SPY, ITB/SPY, XRT/SPY | Sectores que históricamente hacen techo antes (a verificar en F1) | Yahoo | ≈2001–2006 | ↓ | F2 |
| Zweig Breadth Thrust | Señal de reentrada (amplitud de muy baja a muy alta en 10 ruedas) | Derivado | — | reentrada | F3 |
| Hindenburg Omen y similares | Clásicos: se testean con escepticismo | Derivado | — | ± | F3 |
| Dispersión de retornos entre componentes | Mercado de "stock pickers" vs macro | Derivado | — | ± | F2 |
| Concentración: peso del top 10 del S&P 500 | Fragilidad por concentración | Holdings de ETF | reciente | ↑ | F2 |
| Volumen alcista / volumen bajista | Presión de compra o venta agregada | Derivado | según universo | ↓ | F2 |

### Pilar 4: Tendencia y momentum

| Variable | Qué mide / cálculo | Fuente | Historia | Signo | Fase |
|----------|--------------------|--------|----------|-------|------|
| SPY vs 50/200 DMA, pendiente de la 200 DMA | Régimen de tendencia (Faber 2007) | Derivado | 1928 (^GSPC) | ↓ | F2 |
| Drawdown actual y días desde el máximo | Estado dentro del ciclo | Derivado | 1928 | ↑ | F2 |
| Momentum de serie de tiempo 1/3/6/12m | Signo y fuerza de la tendencia (Moskowitz et al. 2012) | Derivado | 1928 | ↓ | F2 |
| Retorno ajustado por volatilidad | Calidad de la tendencia | Derivado | 1928 | ↓ | F2 |
| Overnight vs intradía acumulado | Subas solo de noche con ventas en la sesión = posible distribución | Derivado (OHLC) | ≈1993 | ± | F2 |
| Estructura de máximos y mínimos, rupturas de soportes | Cambio de estructura | Derivado | — | ↑ | F2 |

### Pilar 5: Cross-asset y macro

| Variable | Qué mide / cálculo | Fuente | Historia | Signo | Fase |
|----------|--------------------|--------|----------|-------|------|
| Curva 10a−2a y 10a−3m; **desinversión** | Recesión; la re-empinada después de la inversión suele estar cerca del evento | FRED `T10Y2Y`, `T10Y3M` | 1976 / 1982 | ± | F2 |
| Shock de tasas: Δ del 10a en 21/63d, en σ | Caídas de tipo "tasas" (2022, feb-2018) | FRED `DGS10` | 1962 | ↑ | F2 |
| Tasas reales (10a TIPS) y breakevens | Condiciones financieras reales, inflación esperada | FRED `DFII10`, `T10YIE` | 2003 | ± | F2 |
| Dólar (DXY / UUP): nivel y velocidad | Dólar fuerte y rápido = ajuste de condiciones globales | Yahoo `DX-Y.NYB` | 1971 | ↑ | F2 |
| **USDJPY, AUDJPY** | Desarme de carry trade (ago-2024); AUDJPY es un barómetro clásico de riesgo | Yahoo / FRED | 1971 | ↓ | F2 |
| Cobre/oro | Crecimiento vs miedo | Yahoo (futuros) | ≈2000 | ↓ | F2 |
| Petróleo (WTI): shocks de ±σ | Shocks de oferta o demanda | FRED `DCOILWTICO` | 1986 | ± | F2 |
| Correlación acciones-bonos (SPY vs TLT, 63d) | Si es positiva, los bonos **no** cubren (2022) | Derivado | 2002 | régimen | F2 |
| Bitcoin | Apetito de riesgo 24/7; los movimientos del fin de semana anticipan el lunes (a testear) | Yahoo `BTC-USD` | 2014 | ↓ | F2 |
| Pedidos iniciales de desempleo; regla de Sahm | Deterioro laboral | FRED `ICSA`, `SAHMREALTIME` | 1967 / 1959 | ↑ | F2 |
| Recesiones NBER | Etiqueta para análisis (no se usa como predictor: se publica con mucha demora) | FRED `USREC` | 1854 | — | F1 |
| Valuación: CAPE, prima de riesgo (earnings yield − tasa real) | Fragilidad lenta; mala para timing, útil como contexto | Shiller | 1871 | ↑ | F2 |

### Pilar 6: Global

| Variable | Qué mide / cálculo | Fuente | Historia | Signo | Fase |
|----------|--------------------|--------|----------|-------|------|
| Retorno overnight de Asia y Europa antes de la apertura de EE. UU. | Nikkei, Hang Seng, Kospi, Taiwán, DAX, Euro Stoxx | Yahoo (índices) | 1965–2007 | ↓ | F2 |
| % de mercados del mundo sobre su 200 DMA | Amplitud global (≈25 ETFs país) | Derivado | 1996 | ↓ | F2 |
| EEM/SPY, EFA/SPY | Liderazgo global vs EE. UU. | Yahoo | 2003 / 2001 | ± | F2 |
| China: FXI, KWEB, USDCNH | Estrés chino (2015) | Yahoo | 2004 / 2013 | ↓ | F2 |
| Monedas emergentes (CEW) | Apetito de riesgo EM | Yahoo | ≈2009 | ↓ | F2 |
| Mercados que lideran techos (Kospi y Taiwán por semis, DAX por cíclicos) | A verificar en F1 con lead-lag | Yahoo | — | ↓ | F2 |

### Pilar 7: Flujo institucional y posicionamiento

| Variable | Qué mide / cálculo | Fuente | Historia | Lag | Fase |
|----------|--------------------|--------|----------|-----|------|
| COT TFF: posición neta de asset managers y leveraged funds en ES, NQ, RTY, VX, bonos, yen; percentil 3 años | Posicionamiento extremo, crowding | CFTC | 2006 (legacy 1986) | datos del martes, publicados el viernes | F5 |
| Short volume fuera de bolsa → **índice tipo DIX** | Ratio de short volume en dark pools ponderado por dólares. Alto suele indicar market makers vendiendo a compradores institucionales (a testear) | FINRA Reg SHO diario | ≈2009 | T+1 | F5 |
| Volumen ATS (dark pools) por ticker | Participación institucional por activo | FINRA OTC Transparency | 2014 | 2–4 semanas | F5 |
| **Días de distribución / acumulación** | Índice cae ≥ 0,2 % con más volumen que el día anterior; conteo en 25 ruedas | Derivado | 1993 | — | F2 |
| Volumen relativo, OBV, Chaikin Money Flow, línea A/D en ETFs de sector | Divergencias precio-volumen, acumulación y distribución | Derivado | — | — | F2 |
| Ubicación del cierre en el rango (CLV) | Cierres débiles repetidos = venta en la sesión | Derivado | — | — | F2 |
| Flujos de ETFs (cambio de acciones en circulación) | Entrada o salida de dinero por sector | Mayormente pago / scrape | — | T+1 | F5 |
| Insiders (Form 4): ratio compras/ventas agregado, compras en racimo | Insiders comprando en pisos | SEC EDGAR | 2003 | 2 días hábiles | F5 |
| 13F: concentración de hedge funds | Crowding (lento) | SEC EDGAR | 1999 | 45 días | F5 |
| Margin debt | Apalancamiento minorista | FINRA (mensual) | ≈1997 | ≈3 semanas | F5 |
| **Flujos mecánicos estimados**: | | Derivado (precios) | | | F5 |
| · Fondos de control de volatilidad | Exposición ∝ vol objetivo / vol realizada; salto de vol = venta forzada | | 1990 | — | |
| · CTAs (trend-following) | Posición estimada con señales de 20/60/120/250d; **distancia a niveles gatillo** | | 1928 | — | |
| · ETFs apalancados | Rebalanceo de fin de día ∝ (L² − L) × retorno × AUM | | 2006 | — | |
| · Paridad de riesgo | Desapalancamiento cuando suben juntas la vol de acciones y la de bonos | | 2002 | — | |
| · Ventana de recompras (blackout) | % del S&P en blackout, estimado desde el calendario de resultados | | — | — | |
| · Rebalanceo de fin de trimestre de pensiones | Estimado por el desempeño relativo acciones/bonos del trimestre | | — | — | |
| Opciones: flujo inusual, bloques, GEX propio | Posicionamiento en opciones | **Pago** (ORATS, CBOE DataShop, etc.) | — | — | opcional |

### Pilar 8: Sentimiento (contrarian en extremos)

| Variable | Fuente | Historia | Lag | Fase |
|----------|--------|----------|-----|------|
| AAII bull − bear | AAII | 1987 | semanal (jueves) | F2 |
| NAAIM exposure | NAAIM | 2006 | semanal | F2 |
| CNN Fear & Greed | scrape, no oficial | ≈2011 | diario | F2 |
| Google Trends ("stock market crash", "recession") | Google | 2004 | semanal | F5 |
| Investors Intelligence | pago | 1963 | semanal | opcional |

### Pilar 9: Estructura de correlación

| Variable | Qué mide / cálculo | Fuente | Signo | Fase |
|----------|--------------------|--------|-------|------|
| Correlación promedio entre sectores (21/63d) | Cuando todo se mueve junto, la diversificación desaparece | Derivado | ↑ | F2 |
| **Absorption ratio** | % de varianza explicado por los primeros k componentes principales; subidas bruscas preceden caídas (Kritzman et al. 2011) | Derivado | ↑ | F2 |
| **Índice de turbulencia** | Distancia de Mahalanobis de los retornos de hoy vs su historia: movimientos inusuales entre activos (Kritzman y Li 2010) | Derivado | ↑ | F2 |
| Número efectivo de apuestas / ratio de diversificación | Diversificación real de un universo o canasta | Derivado | ↓ | F2 |
| Clusters jerárquicos / árbol de expansión mínima | Cómo se reagrupan los activos; fusiones de clusters = contagio | Derivado | ± | F2 |
| Correlación condicional dinámica (DCC-GARCH) | Correlación que reacciona rápido | Derivado | ↑ | F6 |
| Co-excedencias de cola | Probabilidad de caer juntos en días extremos | Derivado | ↑ | F1 |

### Pilar 10: Calendario (modulador, no señal)

| Variable | Uso |
|----------|-----|
| FOMC, CPI, empleo (NFP), resultados de mega caps | Mayor volatilidad esperada: el sensor contextualiza, no alerta |
| OPEX mensual y trimestral, vencimiento del VIX | Cambios de gamma y "pinning" |
| Fin de mes y trimestre, primeros días del mes | Flujos de rebalanceo y aportes |
| Estacionalidad mensual, ciclo presidencial, feriados | Contexto débil; se testea antes de usar |

---

## B. Métricas de comportamiento (por activo o canasta vs una referencia)

| Métrica | Definición | Ventanas |
|---------|------------|----------|
| Beta total | cov(r_a, r_m) / var(r_m) | 63 / 252 / 756d y episodio |
| Beta bajista / alcista | Beta calculada solo con días en que r_m < 0 (o > 0) | 252 / 756d, todo el historial |
| Asimetría de beta | beta bajista − beta alcista | idem |
| Captura bajista / alcista | Retorno compuesto del activo / del mercado, en períodos (meses) de mercado negativo o positivo | todo el historial, 3a, 5a |
| Correlación condicional | Correlación en días de estrés (r_m < −1σ, VIX > p80) vs días de calma | todo el historial |
| MES | Retorno medio del activo en el peor 5 % de días del mercado | todo el historial, 3a |
| Co-excedencia de cola | P(activo en su peor 5 % \| mercado en su peor 5 %) | todo el historial |
| Por episodio | Retorno pico→valle del mercado; retorno relativo; drawdown propio; días a su piso vs piso del mercado; días de recuperación; retorno 63d antes del pico | cada episodio |
| Tasa de acierto defensivo | % de episodios con retorno relativo > 0 | por tamaño y tipo de caída |
| Lead-lag de techos y pisos | Fecha del techo o piso propio − fecha del del mercado; correlación cruzada por rezagos | cada episodio |
| Rebote | Retorno 21/63d desde el piso del mercado | cada episodio |
| RRG | RS-Ratio y RS-Momentum normalizados; cuadrante actual | semanal |
| Fuerza relativa multi-horizonte | Percentil del retorno relativo 1/3/6/12m dentro del universo | diario |
| **Puntaje Refugio** | Combinación de captura bajista, MES, tasa de acierto defensivo y correlación en estrés, condicionada al tipo de caída | — |
| **Puntaje Rebote** | Combinación de retorno post-piso, beta alcista y lead de piso | — |

---

## C. Plantilla de ficha por indicador (`docs/fichas/<id>.md`)

```yaml
id: vix_term_structure
nombre: Estructura temporal del VIX (VIX/VIX3M)
pilar: volatilidad
formula: VIX_close / VIX3M_close
fuente: CBOE (respaldo: Yahoo ^VIX, ^VIX3M)
historia_desde: 2007-12
frecuencia: diaria
lag_publicacion: 0 (disponible al cierre)
normalizacion: percentil expansivo, mínimo 504 ruedas
signo_riesgo: "+"            # más alto = más riesgo
hipotesis: >
  La inversión (>1) indica demanda urgente de protección de corto plazo: estrés agudo.
  La vuelta a <1 después de una inversión indica normalización (reentrada).
referencias: []
estado: candidato           # candidato | implementado | validado | descartado
resultados_validacion: null  # se completa en F3: AUC, anticipación, % tiempo en alerta, aporte incremental
```

---

## D. Proxies propios para datos pagos

Decisión del proyecto: **solo datos gratuitos**. Lo que no existe gratis se construye con fórmulas propias, o se empieza a acumular con snapshots diarios desde el primer día. Estos proxies son aproximaciones: cada uno se valida contra la versión original cuando haya algún tramo gratuito para comparar.

| Dato pago o no disponible | Proxy propio | Cómo se calcula | Historia |
|---------------------------|--------------|-----------------|----------|
| Amplitud NYSE (A/D, % sobre medias, máximos-mínimos) | **Amplitud sobre universo propio** | Mismas fórmulas sobre ETFs de sector, industria y país (sin sesgo), 49 industrias Fama-French (sin sesgo) y S&P 500 actual (con sesgo, marcado) | 1926 (FF), 1998 (ETFs) |
| Put/Call de CBOE (el CSV gratuito termina en 2019) | **Put/Call propio** | Snapshot diario de las cadenas de opciones de SPY, QQQ e IWM (Yahoo): volumen y open interest de puts vs calls | desde el primer snapshot |
| GEX / gamma de dealers | **GEX propio** | Σ gamma (Black-Scholes, con la IV de cada contrato) × OI × 100 × S² × 1 %, con calls en positivo y puts en negativo (convención estándar) | desde el primer snapshot |
| Skew de volatilidad | **Skew propio** | IV de puts ≈5 % fuera del dinero − IV at-the-money, vencimiento más cercano a 30 días | desde el primer snapshot |
| DIX (SqueezeMetrics) | **Índice tipo DIX** | FINRA Reg SHO diario: short volume / volumen total fuera de bolsa, ponderado por dólares sobre los componentes | ≈2009 |
| Posicionamiento de CTAs | **CTA estimado** | Promedio de señales de tendencia (signo del retorno en 20/60/120/250 ruedas), escalado por volatilidad inversa. **Nivel gatillo** = precio al que cambia cada señal | 1928 |
| Exposición de fondos de control de volatilidad | **Vol-control estimado** | Exposición = min(1,5; 10 % / vol realizada 21–63d). La variación diaria de la exposición es el flujo estimado | 1928 |
| Flujos de ETFs | **Δ acciones en circulación** | Snapshot diario de acciones en circulación y activos de cada ETF (Yahoo) | desde el primer snapshot |
| OAS high yield con historia larga | **Spread de crédito compuesto** | BAA10Y (FRED, 1986) + retorno relativo HYG/IEF (2007) | 1986 |
| CNN Fear & Greed | **Miedo/codicia propio** | Los 7 componentes replicables: SPY vs media de 125, máximos-mínimos propios, McClellan propio, put/call propio, VIX vs su media de 50, SPY vs TLT a 20d, HYG vs IEF | 2007 (sin put/call, antes) |
| Correlación implícita (CBOE) | **Correlación realizada promedio** + comparación VIX vs vol de sectores | Correlación media de pares entre sectores; ratio VIX² / Σ w² σ² sectoriales | 1998 |
| Ventana de recompras | **Blackout estimado** | Fechas de resultados (Yahoo): desde ≈30 días antes hasta 2 días después; % del universo en blackout | reciente |
| Indicadores de régimen pagos | **HMM propio** | Modelo oculto de Markov sobre retorno y volatilidad de SPY | 1928 |
