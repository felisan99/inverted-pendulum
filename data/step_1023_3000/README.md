# STEP_1023_3000

Pulso de 3000 ms a voltaje máximo (PWM 1023, 12 V) aplicado al motor con el péndulo colgando, duración total 12 s. El escalón largo hace que el brazo alcance velocidad de régimen durante el forzamiento, a diferencia del impulso breve del experimento de 100 ms. El propósito es validar la trayectoria del modelo (brazo más péndulo) durante el forzamiento prolongado, no solo la oscilación libre posterior.

## Config

- Config: `configs/sim_to_real_validation_3000.toml`
- Modelo: `models/pendulum_high_quality.xml`
- Rm: 5.0 Ω (midpoint entre 4.808 Ω @ 5V y 5.5 Ω @ 12V)
- `armature` (inercia reflejada del rotor): 9.080e-3 kg·m² (J_total menos J1,motor, Rm = 5.0)
- `joint1_damping` (b_arm): 2.059e-3 N·m·s/rad (ensayo de escalón sin péndulo)
- `encoder_damping` (b₂): 3.00e-4 N·m·s/rad (zeta = 0.0131, log-decrement sobre hardware real)
- Duración total: 12 s a 1 kHz
- Datos reales de referencia: pendulo largo + attachment (m2 = 0.034 kg, l2 = 0.2149 m)

## Validación

La comparación sim-vs-real y el análisis de parámetros se documentan en el repo de la tesis (`notes/experimento-validacion-sim-real/`). Aquí solo se genera la salida de la simulación.

Resultado con el modelo Rm = 5.0: omega_n error 2.50 % frente al hardware real. El log-decrement de zeta no es válido en este experimento por la no linealidad de amplitud (el péndulo arranca la fase libre con 57 a 73 grados de deflexión); el valor de esta corrida es la comparación de la trayectoria del brazo durante el forzamiento prolongado.

## Archivos

- `step_1023_3000_sim.csv`: salida de la simulación.
- `step_1023_3000_sim.png`: plot de posición del péndulo.
