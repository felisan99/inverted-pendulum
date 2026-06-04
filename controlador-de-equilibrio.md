# Plan: verificar y corregir el controlador PID de equilibrio (pid_balance.py)

## Contexto

`controllers/pid_balance.py` es un controlador de realimentacion de 4 estados (PID del
pendulo + posicion/velocidad del brazo) para equilibrar el pendulo de Furuta simulado, el
mismo modelo validado contra hardware real para el estudio sim-to-real. Se carga desde el GUI
(`scripts/gui_monitor.py`), que necesita pantalla y por lo tanto no se puede ejecutar de forma
automatica. El objetivo fue verificar si realmente equilibra y, si no, corregir las ganancias
apoyandose en la teoria del proyecto (analisis LQR en `/Users/felipe/Documents/Tesis/Informe-Final`).

Convenciones (confirmadas con `gym_envs/pendulum_sim.py` y `test/test_pendulum_sim.py`):
`pend_enc = 0` es upright, `pend_enc = 2048` es colgando; el controlador recibe `phi =
pend_enc * 2pi/4096` envuelto a `[-pi, pi]` = desviacion del upright.

## Diagnostico

Se creo un arnes headless (`scripts/validate_controller.py`) que replica el camino del GUI
(mismo loader importlib, `reset(initial_angle_rad=...)`, `compute(pend_enc, motor_enc, t_us)`
a 1 kHz) sin pantalla, y mide max|phi|, |phi| final, deriva del brazo, saturacion y un
chequeo de signo de planta.

Resultado con las ganancias originales: **el pendulo se cae** (max|phi| = 180 grados, brazo
deriva a -566 grados). El chequeo de direccion mostro que la realimentacion negativa sobre phi
si estabiliza el canal del pendulo, asi que el problema eran las ganancias.

Comparacion con el LQR validado (`notes/analisis-lqr/`, estado `x = [theta1, phi, theta1_dot,
phi_dot]`, ley `u = -K x`):

| Termino | original | LQR (correcto) | Problema original |
|---|---|---|---|
| Kp / K_phi | 3500 | 20349.7 | ~6x bajo |
| Kd / K_phi_dot | 300 | 1570.6 | ~5x bajo |
| Ka / K_theta1 | +100 | -1018.6 | signo opuesto y ~10x bajo |
| Kb / K_theta1_dot | +50 | -289.4 | signo opuesto y ~6x bajo |

El actuador del LQR (`K_eff = 0.002482 N.m/PWM`) coincide con el de MuJoCo
(`gainprm * 12 / 1023 = 0.002562`), por lo que las ganancias del LQR son directamente
aplicables a la simulacion. Un barrido de configuraciones confirmo que solo el set del LQR
(con Ka y Kb negativos) equilibra; con los signos del brazo invertidos diverge, y la escala
0.25 del firmware no alcanza a equilibrar en esta sim.

## Cambios realizados

1. **`scripts/validate_controller.py`** (nuevo): arnes headless de validacion/tuneo.
   Flags `--controller`, `--perturb-deg`, `--seconds`, `--gain-scale`. Devuelve PASS/FAIL.

2. **`controllers/pid_balance.py`**: ganancias corregidas a los valores del LQR
   (`Kp = 20349.7`, `Kd = 1570.6`, `Ka = -1018.6`, `Kb = -289.4`, `Ki = 0`). Se corrigio el
   docstring que afirmaba que Ka era una "fuerza restauradora" (es lo opuesto a lo optimo:
   el LQR usa la posicion del brazo activamente, con Ka negativa). Se mantuvo el filtro EMA,
   el anti-windup y el guard de divergencia.

3. **`test/test_pid_balance.py`** (nuevo): test de pytest CI-safe (sin display) que corre el
   controlador en `PendulumSim` desde 5 y 10 grados y verifica que max|phi| queda acotado y
   |phi| final cerca de upright.

4. **`scripts/gui_monitor.py`**: corregido el desfase de 180 grados del plot del pendulo en
   `_on_data` (ahora upright = 0 grados y colgando = +-180, sin discontinuidad en el equilibrio,
   coherente con el titulo "Pendulum angle (upright = 0 grados)").

## Resultado de la verificacion

- `python scripts/validate_controller.py --controller controllers/pid_balance.py` -> PASS
  (perturbacion 5 grados: max|phi| = 5.01, |phi| final = 0.13 grados, brazo +1.9 grados,
  saturacion 0.5%).
- Basin de recuperacion: aprox +-12 grados. A partir de ~15 grados el PWM satura y el pendulo
  no se alcanza a frenar (limitacion fisica, no de tuneo). El default del GUI es 5 grados.
- `pytest test/test_pid_balance.py` -> verde.

## Restriccion

No se commitea ni pushea nada (pedido del usuario).

## Verificacion manual (necesita pantalla)

`python scripts/gui_monitor.py`, cargar `controllers/pid_balance.py`, perturbacion 5 grados,
Start control -> el pendulo se mantiene erguido y el plot muestra ~0 grados en upright.
