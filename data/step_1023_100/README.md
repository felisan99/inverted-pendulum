# STEP_1023_100

Pulso de 100 ms a voltaje máximo (PWM 1023, 12 V) aplicado al motor con el péndulo colgando. Se registra la oscilación libre del péndulo después del pulso.

## Config

- Config: `configs/sim_to_real_validation.toml`
- Modelo: `models/pendulum_high_quality.xml`
- Rm: 5.0 Ω (midpoint entre 4.808 Ω @ 5V y 5.5 Ω @ 12V)
- `encoder_damping` (b₂): 3.01e-4 N·m·s/rad (zeta = 0.01312, identificado por log-decrement sobre hardware real, tesis T13)
- Duración total: 12 s a 1 kHz

## Validación

La comparación sim-vs-real y el análisis de parámetros se documentan en el repo de la tesis (`notes/experimento-validacion-sim-real/`). Aquí solo se genera la salida de la simulación.

## Archivos

- `step_1023_100_sim.csv` — salida de la simulación
- `step_1023_100_sim.png` — plot de posición del péndulo
