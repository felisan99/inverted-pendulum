# Plan: SimConfig + arquitectura sim-to-real para PendulumSim/PendulumEnv

## Contexto

La simulacion actual es casi ideal: modela cuantizacion de encoders y dinamica fisica
validada, pero no modela ruido del AS5600, latencia I2C ni jitter de FreeRTOS. Ademas,
`PendulumEnv` es codigo legacy pensado solo para RL: reimplementa la fisica y la lectura de
sensores en paralelo a `PendulumSim`, con voltaje continuo como accion y la construccion de
observaciones enterrada en `get_observation()`.

El objetivo a futuro es que una politica de RL entrenada en simulacion controle el ESP32 y
el sistema real. Para que ese salto sim-to-real funcione, los dos extremos de la politica
(`observacion → accion`) deben ser reproducibles en hardware:

- **Accion**: el ESP32 emite PWM 10-bit. El mapa voltaje↔PWM es lineal y fijo, asi que la
  fidelidad se logra ruteando la accion del env por el canal cuantizado del backend
  (`PendulumSim.step`), sin cambiar el espacio de accion.
- **Observacion**: el vector `[sin,cos,vel]` se arma desde counts crudos. Si esa
  transformacion se duplica (una en el env, otra en el ESP32) y difieren, aparece una
  brecha sim-to-real silenciosa. Debe vivir en una unica pieza compartida.

## Decisiones de diseño tomadas

1. **Composicion, no herencia.** `step`/`reset` tienen tipos de retorno incompatibles entre
   las dos clases y `PendulumEnv` ya hereda de `gym.Env`. `PendulumEnv` adapta un backend.
2. **Sin rename** de `PendulumSim`/`PendulumEnv` (nombres convencionales, ya referenciados en
   tests/docs). La claridad nueva viene de extraer `ObservationEncoder`.
3. **`PendulumBackend` (Protocol)** define el contrato `PWM in → SensorReading out`.
   `PendulumSim` lo cumple por duck typing; un futuro `HardwareBackend` tambien.
4. **`ObservationEncoder`** (counts → obs) es la pieza portable a hardware: una sola fuente
   de verdad usada en training y en despliegue.
5. **Wrap del pendulo**: solo `equilibrium`. El encoder convierte `pend_enc` a `[-pi, pi]`
   centrado en el upright (como `_pend_to_rad` en `pid_balance.py`), continuo cerca de 0.
   `swing_up` queda como limitacion conocida.
6. **Rendering**: `PendulumSim` es el unico dueño del viewer; se elimina `PendulumEnv.render()`.
7. **Jitter**: reloj acumulado por intervalo (`t += dt_nominal + N(0,sigma)`, siempre
   creciente). La velocidad se calcula con el `dt` medido, asi RL y controlador lo sienten.

## Arquitectura objetivo

```
gym_envs/backend.py      PendulumBackend (Protocol) + SensorReading
   ├─ PendulumSim         MuJoCo + ruido + latencia + reloj jitter   (gym_envs/pendulum_sim.py)
   └─ HardwareBackend     link al ESP32 real                          (futuro, fuera de scope)

gym_envs/observation.py  ObservationEncoder: counts → [sin,cos,vel]  (compartido sim/deploy)

gym_envs/pendulum_env.py PendulumEnv(gym.Env): backend + encoder + reward + episodios
                         (solo training; en deploy se usa backend+encoder+red, sin el env)
```

## Archivos a crear

| Archivo | Descripcion |
|---|---|
| `gym_envs/backend.py` | `PendulumBackend` (Protocol) + `SensorReading` (tipo compartido) |
| `gym_envs/observation.py` | `ObservationEncoder` (counts → obs), portable a hardware |
| `gym_envs/sim_config.py` | Dataclass `SimConfig` con `from_toml()` |
| `configs/sim_ideal.toml` | Referencia con todos los parametros en cero |
| `configs/sim_realistic.toml` | Valores estimados del hardware real |

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| `gym_envs/pendulum_sim.py` | Cumple `PendulumBackend`; `SimConfig`/`seed`; reloj jitter, ruido, latencia; `SensorReading` se mueve a `backend.py` |
| `gym_envs/pendulum_env.py` | Composicion: backend + `ObservationEncoder`; accion ruteada por PWM; elimina fisica/obs duplicada y `render()` |
| `scripts/gui_monitor.py` | Carga `configs/sim_config.toml` por convencion si existe |
| `agents/trainer.py` | Acepta y pasa `SimConfig` a `PendulumEnv` |

---

## Paso 1 — gym_envs/backend.py

```python
from __future__ import annotations
from collections import namedtuple
from typing import Protocol, runtime_checkable

SensorReading = namedtuple("SensorReading", ["t_us", "motor_enc", "pend_enc"])

@runtime_checkable
class PendulumBackend(Protocol):
    def step(self, pwm: int) -> SensorReading: ...
    def reset(self, *args, **kwargs) -> SensorReading: ...
    def close(self) -> None: ...
```

`SensorReading` pasa a vivir aca (hoy esta en `pendulum_sim.py`). `pendulum_sim` lo importa
desde aca como dependencia normal. Los tests que hacen `from gym_envs.pendulum_sim import
SensorReading` se actualizan para importarlo desde `gym_envs.backend`.

---

## Paso 2 — gym_envs/observation.py

Pieza compartida sim/deploy. Sin estado de MuJoCo: solo counts → obs.

```python
import math
import numpy as np
from gym_envs.backend import SensorReading

_PENDULUM_LSB = 2 * math.pi / 4096
_MOTOR_LSB    = 2 * math.pi / 1716

def _pend_to_rad(pend_enc: int) -> float:
    angle = pend_enc * _PENDULUM_LSB
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return angle

class ObservationEncoder:
    """counts → [sin(motor),cos(motor),vel_motor, sin(pend),cos(pend),vel_pend].
    Estado: lectura previa para velocidad por diferencia finita. Mismo codigo en
    training y en despliegue sobre ESP32."""

    def reset(self, reading: SensorReading) -> np.ndarray:
        self._prev_motor = reading.motor_enc * _MOTOR_LSB
        self._prev_pend  = _pend_to_rad(reading.pend_enc)
        self._prev_t_us  = reading.t_us
        return self._obs(self._prev_motor, self._prev_pend, 0.0, 0.0)

    def update(self, reading: SensorReading) -> np.ndarray:
        motor = reading.motor_enc * _MOTOR_LSB
        pend  = _pend_to_rad(reading.pend_enc)
        dt    = max((reading.t_us - self._prev_t_us) * 1e-6, 1e-6)
        motor_vel = (motor - self._prev_motor) / dt
        pend_vel  = (pend  - self._prev_pend)  / dt
        self._prev_motor, self._prev_pend, self._prev_t_us = motor, pend, reading.t_us
        return self._obs(motor, pend, motor_vel, pend_vel)

    @staticmethod
    def _obs(motor, pend, motor_vel, pend_vel) -> np.ndarray:
        return np.array([math.sin(motor), math.cos(motor), motor_vel,
                         math.sin(pend),  math.cos(pend),  pend_vel], dtype=np.float32)
```

---

## Paso 3 — gym_envs/sim_config.py

```python
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SimConfig:
    pend_noise_sigma: float = 0.0      # sigma Gaussiano en pend_enc [counts AS5600]
    motor_noise_sigma: float = 0.0     # sigma Gaussiano en motor_enc [counts Hall]
    sensor_latency_steps: int = 0      # pasos de retardo puro en la lectura
    dt_jitter_sigma: float = 0.0       # sigma del jitter POR INTERVALO en t_us [us]

    @classmethod
    def from_toml(cls, path: str | Path) -> SimConfig:
        with Path(path).open("rb") as f:
            data = tomllib.load(f).get("sim_config", {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
```

---

## Paso 4 — gym_envs/pendulum_sim.py

Cumple `PendulumBackend`. Reloj con jitter, ruido y latencia en `_read_sensors()`.
`dt_nominal` se toma del timestep del modelo (no hardcodeado).

**Constructor:**
```python
def __init__(self, xml_file=None, render_mode=None,
             sim_config: SimConfig | None = None, seed: int | None = None):
    ...
    self._config   = sim_config or SimConfig()
    self._rng      = np.random.default_rng(seed)
    self._dt_us    = self._model.opt.timestep * 1e6
    self._clock_us = 0.0
    lat = self._config.sensor_latency_steps
    self._latency_buf: deque[SensorReading] = deque(maxlen=max(lat + 1, 1))
```

**`reset()`** (re-siembra RNG con seed, reinicia reloj y buffer):
```python
def reset(self, pendulum_down=True, initial_angle_rad=None, seed=None) -> SensorReading:
    if seed is not None:
        self._rng = np.random.default_rng(seed)
    self._clock_us = 0.0
    self._latency_buf.clear()
    mujoco.mj_resetData(self._model, self._data)
    ...  # set qpos como hoy
    return self._read_sensors()
```

**`_read_sensors()`:**
```python
def _read_sensors(self) -> SensorReading:
    cfg = self._config
    self._clock_us += (max(self._dt_us + self._rng.normal(0, cfg.dt_jitter_sigma), 1.0)
                       if cfg.dt_jitter_sigma > 0 else self._dt_us)
    t_us = int(self._clock_us)

    motor_enc = int(round(self._data.qpos[0] / _MOTOR_LSB))
    pend_enc  = int(round(self._data.qpos[1] / _PENDULUM_LSB)) % 4096
    if cfg.pend_noise_sigma > 0:
        pend_enc  = (pend_enc + int(round(self._rng.normal(0, cfg.pend_noise_sigma)))) % 4096
    if cfg.motor_noise_sigma > 0:
        motor_enc += int(round(self._rng.normal(0, cfg.motor_noise_sigma)))

    fresh = SensorReading(t_us, motor_enc, pend_enc)
    if cfg.sensor_latency_steps == 0:
        return fresh
    self._latency_buf.append(fresh)
    return self._latency_buf[0]
```

Con `sigma=0` y `latency=0` el comportamiento es identico al actual.

---

## Paso 5 — gym_envs/pendulum_env.py (refactor)

Composicion sobre un `PendulumBackend` + `ObservationEncoder`. Acepta un backend inyectado
(para el futuro `HardwareBackend`); por defecto crea un `PendulumSim`.

**Constructor:**
```python
def __init__(self, xml_file=None, render_mode="human", max_steps=2000,
             task="equilibrium", starting_offset=0.4,
             sim_config: SimConfig | None = None, seed: int | None = None,
             backend: PendulumBackend | None = None):
    ...
    self._backend = backend or PendulumSim(xml_file=xml_file, render_mode=render_mode,
                                           sim_config=sim_config, seed=seed)
    self._encoder = ObservationEncoder()
```

**`reset(seed)`**: `r = self._backend.reset(seed=seed, ...)`; `obs = self._encoder.reset(r)`.

**`step(action)`**: voltaje → PWM (`pwm = round(voltage * 1023/12)`), `r = self._backend.step(pwm)`,
`obs = self._encoder.update(r)`. La accion atraviesa la cuantizacion 10-bit del backend.
**Reward y terminacion sin cambios** (siguen consumiendo el mismo `obs`).

Se elimina la carga de MuJoCo, `get_observation()` duplicada y `render()` (lo hace el backend).

### Limitacion conocida: swing_up
`_pend_to_rad` deja la discontinuidad en ±pi (pendulo colgando), por donde pasa `swing_up`.
La velocidad del pendulo por diferencia finita se rompe en ese cruce. `swing_up` queda fuera
de scope; si se retoma, desenvolver la diferencia (como `angularDistance` del firmware).

---

## Paso 6 — configs/

`configs/sim_ideal.toml`:
```toml
[sim_config]
pend_noise_sigma     = 0.0
motor_noise_sigma    = 0.0
sensor_latency_steps = 0
dt_jitter_sigma      = 0.0
```

`configs/sim_realistic.toml`:
```toml
[sim_config]
pend_noise_sigma     = 1.5    # counts — jitter magnetico AS5600
motor_noise_sigma    = 0.0    # counts — Hall cuadratura es esencialmente exacto
sensor_latency_steps = 1      # 1 paso @ 1 kHz = ~1 ms latencia I2C
dt_jitter_sigma      = 200.0  # us — jitter por intervalo tipico de FreeRTOS
```

El usuario crea `configs/sim_config.toml` (nombre reservado) para que el GUI lo levante.

---

## Paso 7 — scripts/gui_monitor.py

Al construir `PendulumSim` en `SimWorker`:
```python
_SIM_CONFIG_PATH = Path("configs/sim_config.toml")
cfg = SimConfig.from_toml(_SIM_CONFIG_PATH) if _SIM_CONFIG_PATH.exists() else SimConfig()
self._sim = PendulumSim(sim_config=cfg)
```
Sin UI extra. Para cambiar la config, editar el archivo y reiniciar el GUI. El controlador
(pid_balance) sigue recibiendo counts crudos; no usa `ObservationEncoder`.

---

## Paso 8 — agents/trainer.py

`RLTrainer.__init__` acepta `sim_config: SimConfig | None = None` y lo pasa a `PendulumEnv`
en `_make_env()`. El `seed` existente se propaga al env (y al backend) para reproducibilidad
del ruido por episodio.

---

## Verificacion

1. `pytest test/` pasa (tests actualizados a importar `SensorReading` desde `gym_envs.backend`;
   `PendulumSim()`/`PendulumEnv()` sin config = comportamiento ideal/actual).
2. GUI sin `configs/sim_config.toml` → comportamiento actual.
3. Copiar `sim_realistic.toml` a `sim_config.toml`, correr GUI con `pid_balance.py` y verificar
   que estabiliza desde 5 grados (test manual de robustez).
4. Comparar `pid_balance.py` a distintas tasas de control (1 kHz vs 250 Hz, vía `configs/control_config.toml`) bajo la misma config realista.
5. Sanity RL: `PendulumEnv(sim_config=SimConfig())` con seed fijo; comparar una corrida corta
   contra la implementacion previa para confirmar que las observaciones de equilibrium coinciden.
6. Sanity sim-to-real: confirmar que `ObservationEncoder` produce el mismo vector dado el mismo
   `SensorReading`, independiente de si viene de `PendulumSim` o de un backend mock.
