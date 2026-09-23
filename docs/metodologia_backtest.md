# Metodología de backtesting y validación

Objetivo: **backtestear lo máximo posible sin engañarse**. Con pocas caídas grandes en la historia es muy fácil encontrar reglas que "hubieran funcionado" por azar. Este documento fija las reglas **antes** de mirar los resultados.

---

## 1. Qué es una "caída" (eventos)

Todas las definiciones son parametrizables y se aplican a cualquier referencia: SPY, QQQ, IWM, EFA, EEM, ^GSPC, una canasta propia.

| Evento | Definición por defecto | Uso |
|--------|------------------------|-----|
| Episodio de drawdown | Caída ≥ 5 %, 10 % o 20 % desde el máximo previo (cierres). Pico = último máximo antes de cruzar el umbral; valle = mínimo antes de recuperar el pico | Mapa de comportamiento, objetivos del sensor |
| Shock diario | Retorno ≤ −2 %, o ≤ −2σ de la volatilidad de 63d | Muestra grande de "días malos" |
| Spike de volatilidad | VIX +30 % en 5 ruedas, cruce de 30, o VIX/VIX3M > 1 | Estrés agudo |
| Fases del episodio | Pre-pico (−63 ruedas), caída, capitulación (último 10 % del recorrido), rebote (+21), recuperación | Comportamiento por fase |

**Tipo de caída**: se define por la huella macro durante el episodio (Δ tasa 10a, Δ dólar, Δ petróleo, Δ spread de crédito, Δ USDJPY, sector líder de la baja). La agrupación se hace con reglas simples y transparentes (p. ej. "tasas" si Δ10a > +1σ y el crédito estable), no con clustering opaco, porque son pocos episodios.

---

## 2. Objetivos a pronosticar (etiquetas)

Horizonte **swing**: los objetivos principales son a 5, 10 y 21 ruedas.

| Id | Objetivo | Horizonte | Tipo | Rol |
|----|----------|-----------|------|-----|
| Y1 | Caída máxima de SPY ≥ 3 % dentro de las próximas 5 ruedas | 5d | binario | estrés inmediato |
| Y2 | Caída máxima de SPY ≥ 5 % dentro de las próximas 10 ruedas | 10d | binario | principal |
| Y3 | Caída máxima de SPY ≥ 5 % dentro de las próximas 21 ruedas | 21d | binario | **principal** |
| Y4 | Caída máxima de SPY ≥ 10 % dentro de las próximas 63 ruedas | 63d | binario | contexto de fondo |
| Y5 | Volatilidad realizada de las próximas 21 ruedas > percentil 80 histórico (point-in-time) | 21d | binario | riesgo |
| Y6 | VIX cierra > 30 dentro de las próximas 21 ruedas | 21d | binario | estrés |
| Y7 | Retorno de las próximas 21 ruedas en el decil inferior | 21d | binario | cola |
| Y8 | Retorno relativo de cada sector vs SPY en las próximas 10/21 ruedas, **condicionado a que Y3 ocurra** | 10–21d | continuo | rotación |

La "caída máxima dentro de las próximas N ruedas" se mide desde el cierre de *t*: min(cierre_{t+1..t+N}) / cierre_t − 1.

Las etiquetas se **superponen** en el tiempo: con horizonte de 21 ruedas, la etiqueta de hoy y la de mañana comparten 20 ruedas. Esto obliga a usar purga y embargo (§4) y a no tratar las observaciones diarias como independientes.

---

## 3. Reglas anti-sesgo (obligatorias, con tests)

1. **Sin look-ahead.** Una señal calculada al cierre de *t* usa solo datos con `fecha_disponible ≤ t`. La ejecución es en *t+1* (apertura o cierre, configurable).
2. **Demoras de publicación**: COT (dato del martes, publicado el viernes), ATS (2–4 semanas), 13F (45 días), Form 4 (2 días), AAII (jueves), macro mensual (según calendario de publicación).
3. **Revisiones de datos**: las series macro que se revisan (claims, empleo) se toman del *primer dato publicado* (ALFRED), no de la serie revisada.
4. **Normalización solo con el pasado**: percentiles y z-scores con ventana expansiva o móvil. **Nunca** con media o desvío de toda la muestra.
5. **Sesgo de supervivencia**: se prefieren ETFs e índices, que no desaparecen. Todo análisis sobre constituyentes actuales lleva una advertencia explícita.
6. **Precios**: retornos totales (ajustados por dividendos y splits). Se guarda un snapshot versionado, porque los ajustes de Yahoo cambian con el tiempo.
7. **Husos horarios**: el cierre de Asia del día *t* se puede usar para EE. UU. del día *t*. El cierre de EE. UU. del día *t* **no** se puede usar para Asia del día *t*. Europa se trata con cuidado por la superposición horaria.
8. **Composición cambiante**: XLK y XLC se recompusieron en 2018 (GOOGL y META pasaron a XLC) y las mega caps pesan cada vez más. Hay que tenerlo en cuenta al comparar épocas.
9. **Test automático anti look-ahead**: para fechas al azar, recalcular cada indicador con los datos truncados a esa fecha tiene que dar el mismo valor que en la serie completa.

---

## 4. Esquema de validación

### 4.1 Particiones

| Tramo | Período | Uso |
|-------|---------|-----|
| Desarrollo | inicio → 2014 | Diseño, selección de variables, ajuste de parámetros |
| Validación | 2015 → 2019 | Comparar variantes y elegir la configuración |
| **Holdout sellado** | 2020 → hoy | Se evalúa **una sola vez** por familia de modelos, al final. Si se mira y se re-ajusta, deja de ser holdout y se reporta como tal |
| Mercados no usados | EFA, EWJ/Nikkei, DAX, EEM | Pseudo fuera de muestra: mismos parámetros, otro mercado |
| Historia larga | ^GSPC 1928–1998, industrias Fama-French 1926→ | Para indicadores que solo usan precios: ¿el patrón existía antes? |

Caveat honesto: el análisis descriptivo de la Fase 1 mira episodios de 2020 en adelante, lo que "contamina" el holdout de forma débil. Por eso son tan importantes los mercados no usados y la historia larga.

### 4.2 Walk-forward con purga y embargo

- Ventana expansiva, re-entrenamiento anual (o trimestral para modelos con pesos).
- **Purga**: se eliminan del entrenamiento las observaciones cuya etiqueta se superpone con el tramo de prueba.
- **Embargo**: margen de ruedas igual al horizonte de la etiqueta (5, 10, 21 o 63) después de cada tramo de prueba (López de Prado 2018).

### 4.3 Pocos eventos: cómo compensar

- Reportar **siempre** la tabla por episodio, no solo promedios.
- Intervalos de confianza por **bootstrap en bloques**, que respeta la autocorrelación.
- Aumentar la muestra con eventos menores (5 %, shocks), otros mercados e historia larga.
- Fijar el signo esperado de cada variable **antes** de testear. Un resultado con el signo contrario al esperado es sospechoso, no un descubrimiento.

---

## 5. Métricas

### 5.1 Predictivas (sensor)

| Métrica | Por qué |
|---------|---------|
| AUC-ROC y **AUC-PR** | AUC-PR es la relevante con eventos raros |
| Brier score y diagrama de calibración | Que un 30 % signifique 30 % |
| **Anticipación** (días entre la primera alerta y el pico, o el −5 %) | Una alerta que llega tarde no sirve |
| Precisión y recall en el umbral operativo | Cuántas alertas son "verdaderas" |
| Falsas alarmas por año | Costo de convivencia |
| **% del tiempo en alerta** y cambios de estado por año | Presupuesto de alertas |
| Aporte incremental (Δ AUC-PR al agregar la variable al compuesto) | Evita sumar redundancias |

### 5.2 Económicas (overlay y rotación)

CAGR, volatilidad, Sharpe, Sortino, **máximo drawdown**, **Calmar**, Ulcer index, captura alcista y bajista vs benchmark, rotación de cartera y costos, % del tiempo invertido, peor mes y peor año, tiempo de recuperación.

Siempre contra: **buy & hold**, **SPY > 200 DMA**, **volatility targeting simple** y **exposición aleatoria con el mismo % de tiempo invertido** (Monte Carlo). Si una señal no le gana a una exposición al azar con el mismo tiempo afuera, no está aportando timing.

---

## 6. Robustez

| Prueba | Criterio |
|--------|----------|
| Sensibilidad de parámetros (mapas de calor) | Elegir **mesetas**, no picos; resultados estables a ±20 % |
| Subperíodos (por década, por régimen de tasas) | Sin depender de un solo episodio (p. ej. solo 2008) |
| Mercados alternativos | Mismo signo del efecto en al menos la mayoría |
| Costos × 2, demora de ejecución +1 día | La mejora sobrevive |
| Exclusión del mejor episodio | La estrategia sigue siendo mejor que el baseline |
| **Deflated Sharpe Ratio** | Corrige por la cantidad de variantes probadas |
| **Probabilidad de sobreajuste (PBO, CSCV)** | < 0,2 para promover |

**Registro de experimentos**: cada prueba (variable, parámetros, período, resultado) se guarda automáticamente. La cantidad de pruebas alimenta el Deflated Sharpe. Sin registro no hay corrección honesta.

---

## 7. Protocolo "no pisar trades"

El sensor tiene que demostrar que **no destruye** el rendimiento de lo que ya funciona.

1. **Sobre estrategias de referencia**: buy & hold SPY, rotación sectorial por momentum y una cartera de líderes por fuerza relativa. Se aplica el overlay y se mide:
   - Δ máximo drawdown, Δ Calmar.
   - **Captura alcista retenida** (meta ≥ 85–90 %).
   - Cantidad de "subas perdidas": días de +2 % o más estando reducido.
2. **Sobre estrategias swing de referencia**: rupturas de máximos de 20/55 ruedas, pullbacks a la media de 20 en tendencia, momentum sectorial con rebalanceo semanal. Para cada trade del backtest:
   - Estado del sensor al abrir y durante el trade.
   - % de trades ganadores que el overlay habría recortado y P&L resignado.
   - % de trades perdedores evitados o reducidos y P&L ahorrado.
   - **Neto** por estado. Si es negativo en un estado, el overlay no actúa en ese estado.
   - Si más adelante aparece un historial de trades propio (CSV del broker), se aplica el mismo análisis.
3. **Reglas de diseño que se validan**:
   - Reducción gradual en vez de binaria.
   - Los líderes con fuerza relativa en máximos se excluyen del recorte.
   - Histéresis y persistencia mínima.
   - Reentrada explícita: capitulación, breadth thrust, normalización del VIX.

---

## 8. Forward test (validación en vivo)

El backtest siempre tiene algo de sesgo, porque uno conoce la historia. El forward test no lo tiene.

1. **Registro diario inmutable**: cada mañana hábil, antes de la apertura, GitHub Actions guarda (para la última rueda cerrada) en `registro/` los valores de todos los indicadores y del sensor tal como se veían ese día (fecha, hora UTC, versión del código). Nunca se reescribe una fila pasada.
2. **Snapshot de datos**: el registro captura también las revisiones de datos (precios ajustados que cambian, series macro revisadas). Queda como una base point-in-time propia.
3. **Reglas congeladas**: cada versión del sensor tiene un identificador. Las señales en vivo se evalúan solo contra la versión que las generó. Cambiar reglas = nueva versión y nuevo track record.
4. **Evaluación periódica**: cada semana se completan las etiquetas cuyo horizonte ya venció (Y1…Y8). Cada trimestre se comparan las métricas en vivo con las del backtest. Si las métricas en vivo son mucho peores, es una señal de sobreajuste.

## 10. Ponderación de indicadores (implementada)

Los indicadores no pesan igual: pesa más lo que más anticipó caídas. Código: `src/compomercado/indicadores/ponderacion.py`.

1. **Importancia** = AUC del percentil de riesgo de cada indicador contra Y1, Y2 e Y3 (promedio). 0,5 = no informa.
2. **Depuración** dentro de cada pilar: afuera el indicador con menos de 750 ruedas evaluables, el que no informa (AUC < 0,52), el que funciona al revés de lo esperado (AUC < 0,48; no se invierte el signo) y el que repite a otro más importante (correlación ≥ 0,85).
3. **Pesos por ranking**: el más importante pesa n, el siguiente n−1, … el último 1, normalizados. Se eligió ranking y no el AUC crudo porque es más estable y sobreajusta menos.
4. **Entre pilares**: se mide el AUC de cada pilar ya ponderado y se aplica el mismo ranking.
5. **Walk-forward anual**: los pesos de cada año se estiman con datos hasta 21 ruedas antes del 1 de enero (purga igual al horizonte más largo). La serie publicada es la fuera de muestra desde 2005.
6. **Evaluación**: la pestaña Ponderaciones compara, fuera de muestra y por período, el total ponderado contra pesos iguales y contra referencias simples (SPY bajo su media de 200, nivel del VIX).

## 9. Criterios para promover algo a "producción"

Un indicador, modelo o regla pasa a producción solo si cumple todo esto:

- [ ] Ficha completa (definición, fuente, lag, historia, signo esperado, hipótesis).
- [ ] Tests unitarios y test anti look-ahead en verde.
- [ ] Signo del efecto igual al esperado en desarrollo **y** en validación.
- [ ] Aporte incremental positivo fuera de muestra (no solo significativo aislado).
- [ ] Robusto en subperíodos y en al menos un mercado alternativo.
- [ ] Dentro del presupuesto de alertas (tiempo en alerta, cambios por año).
- [ ] Deflated Sharpe / PBO aceptables (para reglas de trading).
- [ ] Resultado registrado en el registro de experimentos.

Una vez en producción, las señales se registran **en vivo y de forma inmutable** (con fecha y hora). Ese track record es la validación definitiva y se revisa cada trimestre.
