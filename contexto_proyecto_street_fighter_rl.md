# Contexto del proyecto — Agente autónomo de Street Fighter II mediante aprendizaje por refuerzo profundo

> **Propósito de este documento:** proporcionar a agentes de programación y asistentes ejecutados en terminal un contexto único, preciso y operativo del proyecto.
>
> **Fuente principal:** presentación del proyecto `Presentación 1 RL (4).pdf` (10 diapositivas).
>
> **Regla de fidelidad:** este documento distingue entre información definida en la presentación y decisiones todavía no especificadas. No conviertas elementos pendientes en hechos ni elijas valores técnicos por cuenta propia. Cuando una tarea dependa de una decisión pendiente, detente y solicita esa definición al usuario antes de implementarla.

---

## 1. Resumen del proyecto

El proyecto busca entrenar un agente autónomo para jugar **Street Fighter II** mediante **aprendizaje por refuerzo profundo** (*Deep Reinforcement Learning*, DRL).

El agente debe aprender a interpretar el estado del combate y seleccionar acciones para pasar de no saber jugar a enfrentarse y vencer a la CPU del videojuego en dificultad media, identificada en la presentación como **dificultad 4**.

El agente aprenderá mediante interacción con el videojuego. No se plantea entregarle una secuencia fija de movimientos ganadores. El aprendizaje se apoya en observaciones del estado, acciones discretas y recompensas relacionadas con el daño, la distancia y el resultado del combate.

### Objetivo general

Desarrollar y entrenar un agente autónomo mediante aprendizaje por refuerzo profundo que aprenda una política de combate en Street Fighter II y consiga superar a la CPU del juego en dificultad 4.

### Meta cuantitativa final indicada

- **Más de 75 % de tasa de victoria** frente a la CPU en dificultad 4.
- Evaluación principal mediante **100 combates independientes**.
- Entrenamiento planificado hasta **1.000.000 de timesteps** (hito S10).

Estas cifras son objetivos de la propuesta, no resultados ya demostrados.

---

## 2. Alcance conocido

### Incluido en la propuesta

1. Utilizar Street Fighter II como entorno de interacción.
2. Conectar el videojuego mediante Stable-Retro y la interfaz Gymnasium.
3. Obtener observaciones a partir de variables internas/RAM del videojuego.
4. Representar el estado con un vector compacto de aproximadamente 12–20 dimensiones.
5. Reducir las combinaciones de controles a un espacio discreto de 10–14 macroacciones.
6. Diseñar una recompensa densa relacionada con variaciones de HP, victoria/derrota y distancia.
7. Priorizar PPO como algoritmo de entrenamiento.
8. Mantener DQN/Rainbow como referencia (*baseline*), sujeto a aclarar la variante concreta.
9. Utilizar Stable-Baselines3, PyTorch y TensorBoard.
10. Utilizar entornos vectorizados con `SubprocVecEnv` para recolectar experiencia en múltiples combates simultáneos.
11. Entrenar contra la CPU en dificultad 4.
12. Evaluar tasa de victoria, diferencial de daño neto y estabilidad de la política.
13. Considerar pruebas de benchmarking en dificultades 1, 4 y 7.

### Fuera de lo que está definido

La presentación no define un sistema de visión por computador basado en píxeles, entrenamiento mediante *self-play*, aprendizaje por imitación, o un oponente neuronal. No incorpores estas técnicas como parte del diseño sin autorización expresa.

Tampoco se especifican todos los detalles de implementación, los hiperparámetros finales, el personaje seleccionado ni la configuración completa de los combates. Consulta la sección **Decisiones pendientes y preguntas obligatorias**.

---

## 3. Formulación como proceso de decisión de Markov (MDP)

La presentación define el MDP como:

\[
\mathcal{M} = \{S, A, P, R, \gamma\}
\]

Donde:

- `S`: espacio de estados u observaciones.
- `A`: espacio de acciones.
- `P`: dinámica de transición del videojuego.
- `R`: función de recompensa.
- `γ`: factor de descuento.

La presentación identifica los cinco componentes, pero no entrega una especificación completa de `P` ni un valor para `γ`.

### 3.1. Espacio de estados (`S`)

El agente recibirá una observación construida a partir de la RAM/variables internas accesibles mediante Stable-Retro.

La presentación menciona explícitamente:

- Coordenadas `(x, y)` de los personajes.
- Vida/HP de ambos jugadores.
- Estado de animación.
- Vector compacto de aproximadamente **12–20 dimensiones**.

La lista final de variables no está definida. No inventes campos ni asumas que las variables mencionadas son la lista completa.

### 3.2. Espacio de acciones (`A`)

La presentación considera **12 controles binarios crudos**, lo que produce:

\[
2^{12} = 4096
\]

combinaciones posibles.

Para reducir el espacio de decisión, se propone mapear las combinaciones de controles a un espacio discreto de **N = 10 a 14 macroacciones esenciales**.

Las categorías de macroacciones mencionadas son:

- Movimiento.
- Bloqueo.
- Golpes débiles y fuertes.
- Patadas.
- Saltos.

La presentación no proporciona el catálogo definitivo ni el mapeo exacto de cada macroacción a los controles del juego. No inventes ese catálogo.

La interfaz del espacio de acciones se expresa como `Discrete(N)`, donde `N` debe quedar dentro del rango propuesto de 10 a 14 y su valor exacto está pendiente.

### 3.3. Transición (`P`)

Las transiciones las produce el propio videojuego al ejecutar las acciones del agente y avanzar el entorno. El agente recibe la nueva observación tras la interacción.

No se ha especificado una función de transición matemática propia ni un modelo predictivo separado del videojuego.

### 3.4. Función de recompensa (`R`)

La presentación propone la siguiente estructura:

\[
R_t = \alpha \cdot \Delta HP_{oponente}
      - \beta \cdot \Delta HP_{agente}
      + R_{victoria/derrota}
      + R_{distancia}
\]

Componentes indicados:

- `ΔHP_oponente`: variación de HP del oponente.
- `ΔHP_agente`: variación de HP del agente.
- `α` y `β`: coeficientes de ponderación.
- `R_victoria/derrota`: componente asociado al resultado del combate.
- `R_distancia`: componente relacionado con la distancia entre los jugadores.

La presentación caracteriza el enfoque como una recompensa densa basada en `ΔHP + distancia` y añade el resultado de victoria/derrota en la ecuación.

**No están definidos:** valores de `α` y `β`, fórmula y escala de `R_distancia`, magnitud de las recompensas terminales, frecuencia de cálculo ni convención exacta de signos para las variaciones de HP. No implementes valores ni convenciones por inferencia.

### 3.5. Factor de descuento (`γ`)

`γ` forma parte de la definición del MDP, pero la presentación no proporciona su valor. Debe tratarse como pendiente.

---

## 4. Algoritmos de aprendizaje

### 4.1. PPO — algoritmo prioritario

**PPO (Proximal Policy Optimization)** es el algoritmo priorizado para entrenar al agente.

La presentación lo considera adecuado para:

- Procesar el espacio de acciones discretizado `Discrete(N)`.
- Aprender una política de combate.
- Regular la exploración mediante el coeficiente de entropía.
- Monitorizar la entropía mediante TensorBoard.

La presentación también asigna al trabajo de ajuste de PPO el *tuning* de entropía y `clip range`.

No están definidos los valores de los hiperparámetros ni la arquitectura completa de la red. La diapositiva de roles sí menciona una **arquitectura MLP**, pero no especifica cantidad de capas, unidades, activaciones ni otras características.

### 4.2. DQN / Rainbow — referencia (*baseline*)

La presentación muestra `DQN / Rainbow` como alternativa y señala que se mantiene como referencia base (*baseline*), mientras se prioriza PPO.

La justificación presentada para priorizar PPO indica que, en juegos de lucha con transiciones rápidas y recompensas densas (`ΔHP + distancia`), DQN puede sufrir sobreestimación de valores Q y requerir un ajuste de hiperparámetros más costoso.

**Ambigüedad pendiente:** la presentación no aclara si el baseline que se implementará será DQN clásico, Rainbow completo u otra variante. No elijas una variante sin consultar.

### 4.3. Comparación prevista

- PPO: algoritmo prioritario.
- DQN/Rainbow: algoritmo de referencia, si se concreta su implementación.
- La presentación no define un protocolo experimental completo para que la comparación sea justa (por ejemplo, presupuesto de pasos, semillas y configuraciones equivalentes). Esos detalles están pendientes.

---

## 5. Entorno de software y herramientas

El *stack* que aparece en la presentación es:

| Componente | Rol indicado en el proyecto |
|---|---|
| Python | Lenguaje de desarrollo previsto para integrar el entorno y el entrenamiento. |
| Stable-Retro / Gym Retro | Acceso al videojuego, variables internas y controles. |
| Gymnasium | Interfaz del entorno de aprendizaje por refuerzo. |
| Stable-Baselines3 | Implementación de algoritmos de RL. |
| PyTorch | Base de las implementaciones de aprendizaje profundo. |
| TensorBoard | Monitoreo de métricas y del entrenamiento. |
| `SubprocVecEnv` | Ejecución paralela de múltiples entornos/procesos. |

La presentación también menciona ROMs y ejecución *headless* como parte de las tareas de infraestructura.

No están especificadas las versiones de las bibliotecas, el sistema operativo, el hardware, la ROM exacta ni los requisitos de instalación. No fijes versiones arbitrarias en archivos de dependencias sin revisar el estado real del repositorio o consultar al usuario.

---

## 6. Protocolo de interacción y entrenamiento

El ciclo conceptual del agente es:

1. El entorno entrega al agente una observación basada en variables internas/RAM.
2. La política del agente selecciona una macroacción del espacio discreto.
3. El entorno ejecuta la acción en Street Fighter II.
4. El entorno avanza y devuelve una nueva observación y la recompensa calculada.
5. PPO utiliza la experiencia recolectada para actualizar la política.
6. El ciclo se repite durante el entrenamiento.

Este ciclo describe el enfoque general; los detalles de duración de acción, frecuencia de observación y reinicio de episodios no están fijados en la presentación.

### Oponente de entrenamiento

El agente entrenará contra la IA interna de Street Fighter II en **dificultad 4**.

### Paralelización

Se propone utilizar `SubprocVecEnv` para recolectar experiencia simultáneamente en múltiples combates.

La presentación no fija el número de entornos paralelos ni la configuración de procesos.

### Presupuesto de entrenamiento

- Hito S10: alcanzar **1.000.000 de timesteps**.
- La gráfica de convergencia ilustra un punto de control en torno a **500.000 timesteps (S4)** y el objetivo de **1.000.000 (S10)**.
- La presentación no explica con precisión si el contador se reportará como pasos agregados entre todos los entornos paralelos o con otra convención. Define y documenta esta convención antes de reportar el presupuesto alcanzado.

Un *timestep* no debe confundirse con un combate completo.

---

## 7. Métricas y evaluación

### 7.1. Tasa de victoria (*Win Rate*)

Métrica principal:

\[
WinRate = \frac{\text{combates ganados}}{N} \times 100
\]

La presentación propone `N = 100` combates independientes contra la CPU en dificultad 4.

Meta final: **más de 75 %**.

### 7.2. Diferencial de daño neto (`ΔHP`)

La presentación define:

\[
\Delta HP = HP_{agente} - HP_{rival}
\]

Se mide al sonar el K.O. y busca medir la contundencia táctica, más allá de ganar por tiempo.

No está definido cómo se tratarán los combates que terminen por tiempo u otros resultados sin K.O. No inventes esa regla.

### 7.3. Estabilidad de la política

Se propone observar en TensorBoard:

- Recompensa episódica media.
- Entropía.

El propósito declarado es evaluar la convergencia y evitar el colapso estocástico de la política.

### 7.4. Referencias de rendimiento esperadas

La presentación muestra los siguientes valores como referencias/expectativas:

| Referencia | Tasa de victoria mostrada |
|---|---:|
| Agente aleatorio | 12 % |
| Hito S4 preliminar | 45 % |
| Hito S10 final | >75 % |

**No son resultados confirmados del entrenamiento.** Deben reportarse como metas o valores ilustrativos de la presentación hasta que existan mediciones reales.

### 7.5. Benchmark por dificultad

La diapositiva de asignación de roles menciona benchmarking en dificultades **1, 4 y 7**, con **100 combates / ΔHP neto**.

La presentación no especifica inequívocamente si son 100 combates en total o 100 por dificultad. No asumas una de las dos interpretaciones; consulta antes de implementar el protocolo.

La dificultad 4 sigue siendo la dificultad principal para entrenar y para medir la meta de Win Rate.

---

## 8. Roles del equipo

La presentación asigna las siguientes responsabilidades:

### Martín González — Infraestructura y cómputo

- Entorno base: Gymnasium, Retro, ROMs y ejecución *headless*.
- Vectorización paralela mediante `SubprocVecEnv`.
- Escalado hasta 1 millón de pasos y *profiling* de FPS.

### Alan Ibacache — Observaciones RAM y política PPO

- Extracción y normalización del vector RAM (12–20 dimensiones).
- Discretización de acciones (`Discrete(N)`) y arquitectura MLP.
- Ajuste de PPO: entropía para mitigar sesgos y `clip range`.

### Javier Torres — *Reward shaping* y benchmark

- Formulación de recompensa densa (`ΔHP + distancia`).
- Monitoreo en TensorBoard y evaluación preliminar S4.
- Benchmark en dificultades 1, 4 y 7 (100 combates / ΔHP neto).

### Hitos globales

- **Hito 1:** resultados preliminares, asociado a la iteración 2 y al punto S4.
- **Hito 2:** entrega final y presentación, asociado a S10.

La diapositiva muestra una planificación dividida en S1–S10, pero no entrega información textual suficiente para reconstruir con certeza todas las fechas, duraciones o dependencias de cada barra del cronograma. No deduzcas un calendario detallado a partir de ese gráfico.

### 8.1. Distribución técnica de módulos en el repositorio (`PROPUESTA` — sujeta a aprobación del equipo)

Para traducir los roles anteriores a código sin generar conflictos de Git entre los integrantes, se propone desacoplar el trabajo en módulos independientes dentro de la estructura del repositorio:

#### 1. Martín González — Infraestructura, entorno base y escalado
- **Carpetas / módulos propuestos:**
  - `src/hadoken_rl/envs/base_env.py`: instanciación de `StreetFighterIISpecialChampionEdition-Genesis-v0` mediante `stable-retro` y `gymnasium`, configuración de ejecución *headless*, carga de estados iniciales (`.state`) y reinicio limpio de combates.
  - `src/hadoken_rl/envs/vec_env.py`: paralelización de combates simultáneos con `SubprocVecEnv`, manejo de semillas por proceso y cierre seguro de subprocesos.
  - `scripts/`: utilidades de importación/verificación local de la ROM (carpeta `roms/` ignorada en Git), pruebas de humo (*smoke tests*) y *profiling* de FPS / escalamiento hacia 1.000.000 de *timesteps*.
  - `configs/`: plantillas de configuración reproducible de ejecución e infraestructura.
- **Entregable para el equipo:** entorno base funcional y vectorizable sobre el cual Alan y Javier puedan conectar sus *wrappers* sin ocuparse de subprocesos, ROM ni renderizado.

#### 2. Alan Ibacache — Observaciones RAM, macroacciones y agente PPO
- **Carpetas / módulos propuestos:**
  - `src/hadoken_rl/envs/observation_wrapper.py` (`gymnasium.ObservationWrapper`): lectura de variables internas/RAM del emulador —incluyendo variables no expuestas por defecto en el `data.json` base de `stable-retro`, como coordenadas `(x, y)` y estado de animación—, validación y normalización del vector de 12–20 dimensiones.
  - `src/hadoken_rl/envs/action_wrapper.py` (`gymnasium.ActionWrapper`): traducción del espacio discreto `Discrete(N)` (`N` entre 10 y 14 macroacciones) hacia los 12 controles binarios del emulador.
  - `src/hadoken_rl/agents/`: construcción de la política PPO con arquitectura MLP en Stable-Baselines3/PyTorch, ajuste de hiperparámetros (`ent_coef`, `clip_range`, etc.) y definición del *baseline* cuando se acuerde su variante.
- **Entregable para el equipo:** *wrappers* de estado (`S`) y acción (`A`) desacoplados, junto con el constructor del agente listo para entrenamiento y evaluación.

#### 3. Javier Torres — *Reward shaping*, monitoreo y *benchmark*
- **Carpetas / módulos propuestos:**
  - `src/hadoken_rl/envs/reward_wrapper.py` (`gymnasium.Wrapper`): implementación de la función de recompensa densa ($R_t = \alpha \cdot \Delta HP_{oponente} - \beta \cdot \Delta HP_{agente} + R_{victoria/derrota} + R_{distancia}$) y registro de métricas de episodio en el diccionario `info`.
  - `src/hadoken_rl/evaluation/`: *callbacks* y monitoreo en TensorBoard (recompensa episódica media y entropía), evaluación preliminar del Hito S4 y protocolo automatizado de *benchmark* (100 combates, *Win Rate* y diferencial de daño neto $\Delta HP = HP_{agente} - HP_{rival}$ en dificultades 1, 4 y 7).
- **Entregable para el equipo:** *wrapper* de recompensa (`R`) independiente y suite de evaluación estandarizada para medir cualquier modelo entrenado.

#### 4. Punto de unión del equipo (sin pisarse archivos)
- El entorno completo se ensambla encadenando *wrappers* estándar de Gymnasium sobre el entorno base:
  `BaseEnv (Martín)` $\rightarrow$ `ActionWrapper (Alan)` $\rightarrow$ `RewardWrapper (Javier)` $\rightarrow$ `ObservationWrapper (Alan)` $\rightarrow$ `SubprocVecEnv (Martín)`.
- Cada integrante valida su propia capa por separado en `tests/` antes de ejecutar entrenamientos largos.

---

## 9. Supuestos y límites que deben respetarse

Los siguientes puntos son condiciones que el proyecto necesita verificar o decisiones que todavía requieren definición. No los trates como hechos comprobados:

1. **Acceso a RAM:** hay que verificar que Stable-Retro exponga las variables internas requeridas para la ROM concreta.
2. **Suficiencia de observaciones:** hay que validar que el vector escogido contenga la información necesaria para tomar decisiones. El rango 12–20 dimensiones es una propuesta, no una lista cerrada.
3. **Suficiencia de macroacciones:** hay que definir y validar las 10–14 macroacciones. La presentación no entrega su mapeo final.
4. **Validez de la recompensa:** hay que implementar y probar la ecuación una vez definidos sus coeficientes, signos y componentes.
5. **Capacidad de PPO:** que PPO alcance la meta es una hipótesis experimental, no un resultado garantizado.
6. **Suficiencia del presupuesto:** el millón de timesteps es el objetivo de entrenamiento indicado; no garantiza por sí solo alcanzar el Win Rate esperado.
7. **Reproducibilidad:** las condiciones de inicio, personajes, ROM, versión del entorno y semillas deben definirse para que la evaluación sea comparable.

No agregues supuestos adicionales al contexto del proyecto sin identificarlos como propuestas y obtener aprobación del usuario.

---

## 10. Decisiones pendientes y preguntas obligatorias

Antes de implementar una parte que dependa de estos puntos, consulta al usuario. No selecciones una opción por defecto.

### Entorno y combate

- ¿Qué versión exacta de Street Fighter II y qué ROM se utilizarán?
- ¿Qué personaje controlará el agente y qué personaje utilizará la CPU?
- ¿Qué configuración de combate se utilizará (rondas, tiempo, lado inicial y demás opciones relevantes)?
- ¿Qué significa exactamente “dificultad 4” en la versión/ROM seleccionada y cómo se configura?

### Observaciones

- ¿Cuál es la lista definitiva de variables RAM que formará el vector de 12–20 dimensiones?
- ¿Cómo se obtendrán y validarán las direcciones/valores RAM?
- ¿Qué tratamiento se aplicará a rangos, normalización y variables no disponibles?

### Acciones

- ¿Cuál será el valor final de `N` (entre 10 y 14)?
- ¿Cuál es el catálogo exacto de macroacciones y su correspondencia con los 12 controles?
- ¿Cuántos frames se mantendrá cada acción antes de la siguiente observación/decisión?
- ¿Cómo se gestionarán controles simultáneos, acciones inválidas o cambios de lado del personaje?

### Recompensa y MDP

- ¿Cuáles serán los valores de `α` y `β`?
- ¿Cómo se define exactamente cada `ΔHP` y su convención de signos?
- ¿Cuál será la fórmula de `R_distancia`?
- ¿Cuáles serán los valores de recompensa por victoria, derrota y otros finales de episodio?
- ¿Qué valor se utilizará para `γ`?
- ¿Habrá otras penalizaciones o recompensas? La presentación no las especifica.

### Algoritmos y redes

- ¿El baseline será DQN clásico, Rainbow completo u otra variante?
- ¿Cuál será la arquitectura exacta de la MLP?
- ¿Qué hiperparámetros de PPO se utilizarán (incluidos learning rate, `n_steps`, batch size, epochs, gamma, GAE lambda, entropy coefficient y clip range)?
- ¿Qué presupuesto de entrenamiento se asignará al baseline para compararlo con PPO?

### Infraestructura y evaluación

- ¿Cuántos entornos paralelos se ejecutarán?
- ¿Qué hardware y sistema operativo están disponibles?
- ¿Qué versiones de Python, Stable-Retro, Gymnasium, Stable-Baselines3 y PyTorch se fijarán?
- ¿El millón de timesteps se contabilizará de forma agregada entre entornos paralelos?
- Para el benchmark en dificultades 1, 4 y 7, ¿son 100 combates por dificultad o 100 en total?
- ¿Cómo se contabilizarán empates, timeouts, reinicios fallidos y combates sin K.O.?
- ¿Qué semillas y condiciones iniciales se usarán en evaluación?

---

## 11. Instrucciones para agentes de terminal

Al trabajar en este repositorio o proyecto, sigue estas reglas:

1. **Lee este archivo antes de proponer o modificar la arquitectura.** Úsalo como contexto del objetivo y diseño acordado.
2. **No inventes decisiones faltantes.** Consulta la sección 10 cuando una tarea dependa de un dato pendiente.
3. **Distingue hechos de propuestas.** Si sugieres un valor, algoritmo, estructura o mejora no definida aquí, etiquétala como propuesta y solicita aprobación antes de incorporarla como decisión del proyecto.
4. **No cambies silenciosamente el alcance.** PPO es el algoritmo prioritario; DQN/Rainbow se plantea como baseline, pero la variante concreta está pendiente.
5. **No presentes metas como resultados.** Win Rate >75 %, 1 millón de timesteps, 12 % aleatorio y 45 % S4 son valores objetivo/referenciales de la presentación, no resultados experimentales confirmados.
6. **No inventes variables RAM, direcciones de memoria, acciones ni recompensas.** Deben derivarse de la ROM y de definiciones aprobadas.
7. **Valida cada interfaz por separado.** Antes de un entrenamiento largo, comprueba que el entorno inicia, observa, ejecuta una acción, calcula la recompensa y reinicia correctamente. Esta es una recomendación de ingeniería, no un resultado de la presentación.
8. **Documenta configuraciones y resultados reales.** Separa claramente configuración planificada, configuración ejecutada y resultados medidos.
9. **Mantén reproducibilidad.** Registra las configuraciones efectivamente usadas y no afirmes reproducibilidad si faltan datos de ejecución.
10. **Cuando detectes contradicciones** entre el repositorio, este documento y la presentación, no elijas arbitrariamente. Expón la contradicción y pide resolución.
11. **No añadas dependencias o versiones inventadas.** Revisa el entorno existente y pregunta si hay que fijar versiones nuevas.
12. **No afirmes que un entrenamiento convergió o que el agente aprendió** sin métricas o artefactos que respalden esa conclusión.

### Convención de documentación recomendada

Al reportar el estado del trabajo, utiliza estas etiquetas:

- `DEFINIDO EN LA PRESENTACIÓN`: aparece explícitamente en la fuente.
- `CONFIRMADO POR EL USUARIO`: decisión posterior confirmada por el usuario.
- `PENDIENTE`: falta una decisión o dato.
- `PROPUESTA`: alternativa sugerida por un agente, todavía no aprobada.
- `IMPLEMENTADO`: existe en el código y se ha verificado.
- `MEDIDO`: existe un resultado experimental con evidencia.

---

## 12. Referencias incluidas en la presentación

La presentación enumera estas referencias:

1. Nichol, A., Pfau, V., Hesse, C., Klimov, O., & Schulman, J. (2018). *Gotta Learn Fast: A New Benchmark for Generalization in RL*. arXiv:1804.03720. Indicada como referencia fundamental de Gym Retro.
2. Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations in PyTorch*. Journal of Machine Learning Research, 22(268), 1–8.
3. Wurman, P. R., et al. (2022). *Outracing champion Gran Turismo drivers with deep reinforcement learning*. Nature, 602(7896), 223–228. Indicada como caso de éxito de DRL en videojuegos competitivos.
4. Towers, M., et al. (2023). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2308.06912. Publicación oficial del ecosistema Farama Gymnasium.

---

## 13. Resumen de una línea

**Entrenar con PPO un agente de Street Fighter II que observe variables internas del juego (12–20 dimensiones), elija entre 10–14 macroacciones discretas, aprenda mediante una recompensa basada en daño/distancia/resultado, y sea evaluado principalmente contra la CPU en dificultad 4 con una meta superior al 75 % de victorias en 100 combates independientes.**
