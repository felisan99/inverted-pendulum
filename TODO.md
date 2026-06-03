# TODO

Cada tarea es independiente y ejecutable por un agente separado.
Las marcadas con (BLOQUEANTE) deben completarse antes de iniciar entrenamiento RL.

---

## Correcciones críticas

### [x] T01 — Corregir paso de voltaje al actuador en `pendulum_env.py` (BLOQUEANTE)

**Archivo**: `gym_envs/pendulum_env.py`

`PendulumEnv.step()` computa el torque manualmente y lo pasa como control:
```python
torque = (K_T / R) * (voltage - K_E * avg_omega)
data.ctrl[0] = torque
```
Esto bypasea el actuador `general` del XML que ya implementa ese cálculo con los parámetros validados (`gainprm=0.2116`, `biasprm=-0.2311`).

Cambios a realizar:
1. Reemplazar el bloque de cálculo de torque por `data.ctrl[0] = voltage`.
2. Eliminar las constantes de clase `R`, `K_T`, `K_E`.
3. Verificar que `get_observation()`, `compute_reward()` y `reset()` no usen esas constantes.

---

### [x] T02 — Eliminar archivos muertos

**Archivos**: `mujoco_sim/simulation.py`, `mujoco_sim/trayectorias.py`, `main.py`

- `simulation.py`: usa `mujoco.viewer.launch_passive` (roto en macOS sin `mjpython`). Reemplazado por `visualizar_step_100.py`.
- `trayectorias.py`: único contenido es `def seno()`. Verificar que nada lo importe y borrar.
- `main.py`: solo importa módulos, no ejecuta nada. Borrar.

Verificar con `grep -r` que ningún archivo activo importa estos módulos antes de eliminarlos.

---

### [x] T03 — Limpiar `agents/ppo_agent.py` y `utils/plotting.py`

**Archivos**: `agents/ppo_agent.py`, `utils/plotting.py`

- `ppo_agent.py` guarda modelos en `models/ppo_agent` (directorio inexistente). Es inconsistente con `trainer.py` que usa `results/run_N/`. Evaluar si tiene sentido mantenerlo; si no, eliminarlo.
- `utils/plotting.py` tiene `one_graph_per_observation()` y `all_observations_in_one_graph()` que guardan en `images/` (no existe). Verificar que no se usen y eliminar esas funciones.

---

## Simplificación del config TOML

### [x] T04 — Eliminar la sección `[density]` del config y del código

**Archivos**: `configs/step_1023_100_sim.toml`, `configs/characterization_step.toml`, `configs/characterization_impulse.toml`, `mujoco_sim/characterize_system.py`

`pendulum_model_v3.xml` usa `<inertial>` explícito en todos los bodies. MuJoCo ignora la densidad de los geoms cuando existe `<inertial>`, por lo que `[density]` no tiene efecto.

Cambios:
1. Eliminar la sección `[density]` de los tres TOMLs.
2. Eliminar `_set_body_geom_density()` de `characterize_system.py`.
3. Eliminar la lectura de `density_cfg` en `build_parametrized_model()`.

---

### [x] T05 — Simplificar los parámetros del motor en el config TOML

**Archivos**: `configs/step_1023_100_sim.toml`, `mujoco_sim/characterize_system.py`

El config actual usa `gear_ratio`, `rotor_inertia_kg_m2` y `gearbox_efficiency` para calcular `gainprm` y `armature` en tiempo de ejecución. Pero el modelo ya está identificado y esos valores son constantes conocidas. La indirección agrega superficie de error sin beneficio.

Cambios:
1. Reemplazar los campos `gear_ratio`, `rotor_inertia_kg_m2`, `gearbox_efficiency` en `[motor]` por `gainprm`, `biasprm` y `armature` directamente.
2. Eliminar `max_current_amps` (no se usa en ningún lugar del código).
3. Actualizar `build_parametrized_model()` para leer los tres valores directos en vez de calcularlos.
4. Eliminar `_set_joint_armature()` y el bloque de cómputo de armature.

---

### [x] T06 — Actualizar o eliminar configs de caracterización obsoletos

**Archivos**: `configs/characterization_step.toml`, `configs/characterization_impulse.toml`

Estos dos configs tienen parámetros del motor incorrectos (R=10 Ω, Kt=0.686, Kv=1.08) que no corresponden al modelo validado. Determinar si siguen siendo útiles como punto de partida para experimentos futuros o si se pueden eliminar. Si se mantienen, actualizarlos con los valores de `step_1023_100_sim.toml`.

---

## Flujo sim-to-real

### [x] T07 — Extraer parámetros del step automáticamente desde el CSV real

**Archivo**: `mujoco_sim/analisis_step_100_comparacion.py`

El script requiere `--config` para saber la duración y amplitud del step, pero esa información ya está en el CSV (columna `pwm`). La función `load_real_csv()` ya lee la columna `pwm`; falta usarla para derivar el input automáticamente.

Cambios:
1. Agregar función `extract_step_params_from_csv(pwm, t)` que devuelva `(amplitude_voltage, start_time, duration)`. El voltaje se convierte desde PWM con la escala lineal `V = pwm * 12.0 / 1023`.
2. Hacer `--config` opcional: si no se pasa, se usan parámetros validados por defecto (del modelo `v3`) y la señal de entrada se extrae del CSV.
3. Actualizar el docstring de uso del script.

---

### [x] T08 — Agregar modo headless al script de comparación

**Archivo**: `mujoco_sim/analisis_step_100_comparacion.py`

El script actualmente siempre intenta mostrar o guardar la gráfica. Agregar un flag `--output <ruta.png>` que permita correrlo sin mostrar nada (útil para pipelines automatizados). Si no se pasa `--output`, comportamiento actual (guardar en `results/characterization/`).

---

## Entorno RL

### [x] T09 — Verificar consistencia del timestep entre el entorno RL y el modelo validado

**Archivo**: `gym_envs/pendulum_env.py`

El modelo `pendulum_model_v3.xml` no tiene un timestep fijo declarado en el XML; el TOML de sim-to-real lo fija en 1 ms. El entorno RL carga el XML sin fijar el timestep explícitamente, por lo que usa el default de MuJoCo (2 ms).

Verificar:
1. Cuál es el timestep efectivo que usa el entorno RL al cargar `pendulum_model_v3.xml`.
2. Si la estimación de velocidad por diferencia finita en `get_observation()` es coherente con ese timestep.
3. Si corresponde fijar el timestep en el XML o en el constructor del entorno para que coincida con el hardware (1 ms).

---

### [x] T10 — Mover scripts de validación manual fuera de `test/`

**Archivos**: `test/random_episode_test.py`, `test/max_voltage_test.py`

Estos scripts no son tests de pytest (no tienen asserts). Son entrypoints de validación visual. Moverlos a `scripts/` para que `test/` quede reservado para tests reales con pytest.

Actualizar referencias en `CLAUDE.md` y `AGENTS.md`.

---

## Documentación

### [x] T11 — Escribir `README.md`

El archivo actual es un placeholder. Escribir con:
- Descripción del proyecto (péndulo de Furuta, RL + sim-to-real).
- Instalación (`uv sync`, `source .venv/bin/activate`).
- Cómo correr una comparación sim-to-real desde un CSV (`analisis_step_100_comparacion.py`).
- Cómo correr el visualizador 3D (`visualizar_step_100.py --speed 3.0`).
- Cómo entrenar un agente (`trainer.py`).
- Cómo correr un modelo guardado (`predict.py`).
- Estado del modelo: `pendulum_model_v3.xml`, validado junio 2025, error ωn < 0.1%, ζ < 1.1%.

---

## Dependencias entre tareas

- T01 debe completarse antes de cualquier entrenamiento RL.
- T04 y T05 son independientes entre sí pero conviene hacerlas juntas.
- T07 depende de que T04 y T05 estén hechas (para saber qué espera el config simplificado).
- T09 depende de T01 (para evaluar el entorno con el modelo de motor correcto).
- T11 debe escribirse después de T07 (para documentar el flujo simplificado).
