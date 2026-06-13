# STEP_1023_100

Pulso de 100 ms a voltaje máximo (PWM 1023, 12 V) aplicado al motor con el péndulo colgando. Se registra la oscilación libre del péndulo después del pulso.

## Config

- Config: `configs/sim_to_real_validation.toml`
- Modelo: `models/pendulum_high_quality.xml`
- Rm: 5.0 Ω (midpoint entre 4.808 Ω @ 5V y 5.5 Ω @ 12V)
- `armature` (inercia reflejada del rotor): 9.080e-3 kg·m² (J_total menos J1,motor, Rm = 5.0)
- `joint1_damping` (b_arm): 2.059e-3 N·m·s/rad (ensayo de escalón sin péndulo)
- `encoder_damping` (b₂): 3.00e-4 N·m·s/rad (zeta = 0.0131, log-decrement sobre hardware real)
- Duración total: 12 s a 1 kHz

## Validación

La comparación sim-vs-real y el análisis de parámetros se documentan en el repo de la tesis (`notes/experimento-validacion-sim-real/`). Aquí solo se genera la salida de la simulación.

Resultado con el modelo Rm = 5.0: omega_n error 0.06 %, zeta error 0.70 % frente al hardware real.

## Archivos

- `step_1023_100_sim.csv`: salida de la simulación.
- `step_1023_100_sim.png`: plot de posición del péndulo.
