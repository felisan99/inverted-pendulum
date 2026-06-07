# Plan: eliminar codigo legacy y referencias obsoletas

## Contexto

Tras el gran refactor sim-to-real (`fda0bf0 init refactor`) y la limpieza posterior
(`6653b97 delete workflows`, borrado de `pid_balance_4ms.py`), quedaron en el repo restos
legacy de tres tipos: (1) codigo que importaba modulos ya borrados y crasheaba, (2)
documentos de diseno cuyo trabajo ya esta implementado, y (3) referencias en docs a
archivos que ya no existen. Este plan deja el arbol consistente: que `train()` no crashee,
que el experimento de validacion sim-to-real vuelva a ser ejecutable, y que la
documentacion apunte solo a archivos reales.

## A. Codigo muerto eliminado

### `agents/trainer.py`
- Quitado `DQN` del import de `stable_baselines3` (solo PPO/SAC/A2C estan registrados).
- Quitado `from utils.plotting import plot_monitor_data` (el modulo `utils/` fue borrado
  en `fda0bf0`, lo que hacia crashear `train()` al guardar resultados).
- Eliminado el metodo `_save_results()` y su llamada (su unico cuerpo eran las dos
  llamadas a `plot_monitor_data`). Los `train_monitor.csv` / `val_monitor.csv` los sigue
  escribiendo el wrapper `Monitor`; solo se pierden los PNG de curva de aprendizaje.

## B. Config de validacion sim-to-real recreado

### `configs/sim_to_real_validation.toml` (renombrado y recortado)
Antes `step_1023_100_sim.toml`. Se eliminaron las secciones `[density]` (nunca leida),
`[motor]` y `[friction]` (constantes fisicas; viven ahora como `_MOTOR_DEFAULTS` /
`_FRICTION_DEFAULTS` en `mujoco_sim/characterize_system.py`).
`joint1_frictionloss=0.02` se mantiene en los defaults del codigo (critico, no cambiar
sin re-validar).

Validado: la comparacion sim-vs-real reproduce los valores documentados
(omega_n error 0.09%, zeta error 1.07%).

## C. Documentos de diseno obsoletos

- Eliminado `controlador-de-equilibrio.md` (reporte del refactor de `pid_balance`, ya
  implementado y cubierto por `test/test_pid_balance.py`).
- Este `PLAN.md` reemplaza el documento de diseno del refactor sim-to-real, que estaba
  100% implementado (`gym_envs/backend.py`, `observation.py`, `sim_config.py`, configs, GUI).

## D. Referencias obsoletas corregidas

- `CLAUDE.md`: comando de characterization apunta a `sim_to_real_validation.toml`; quitado el
  bullet `utils/plotting.py`; quitado `threshold_example.py` de los ejemplos; corregida la
  afirmacion de CI (los workflows se borraron a proposito; los tests corren localmente con
  `pytest test/`).
- `README.md`: quitado `threshold_example.py` del listado de `controllers/`.
- `configs/CLAUDE.md`: quitado `characterization_step.toml` (nunca existio).
- `docs/custom-controller-tutorial.md`: quitadas las referencias a
  `controllers/threshold_example.py` (el snippet queda como ejemplo inline).

## Verificacion

1. `python -c "import agents.trainer"` sin error.
2. `pytest test/` verde.
3. `python mujoco_sim/analisis_step_100_comparacion.py --config configs/sim_to_real_validation.toml --real <csv_real>`
   reproduce omega_n ~6.25 / zeta ~0.0115.
4. `grep -rn "utils\.plotting\|threshold_example\|characterization_step.toml\|workflows/tests"`
   no devuelve referencias colgadas (codigo y docs).
