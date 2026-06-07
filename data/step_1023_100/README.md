# STEP_1023_100

Pulso de 100 ms a voltaje máximo (PWM 1023, 12 V) aplicado al motor con el péndulo colgando. Se registra la oscilación libre del péndulo después del pulso.

## Config

- Config: `configs/sim_to_real_validation.toml`
- Modelo: `models/pendulum_high_quality.xml`
- Rm: 5.0 Ω (midpoint entre 4.808 Ω @ 5V y 5.5 Ω @ 12V)
- Duración total: 12 s a 1 kHz

## Resultado

Referencia validada contra hardware real (junio 2025):

| Métrica | Real | Sim | Error |
|---------|------|-----|-------|
| ωn (rad/s) | 6.261 | 6.255 | 0.09% |
| ζ | 0.01140 | 0.01152 | 1.07% |
| Relación de amplitud | 1.0 | ~0.94 | ~6% |

## Archivos

- `step_1023_100_sim.csv` — salida de la simulación
- `step_1023_100_sim.png` — plot de posición del péndulo
