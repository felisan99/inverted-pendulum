#!/usr/bin/env python3
"""
visualize_step_response.py

Muestra la simulacion STEP_1023_100 en ventana interactiva de MuJoCo
a velocidad configurable. Solo visualiza el comportamiento del modelo;
la comparacion sim-vs-real y todo el analisis viven en el repo de la tesis.

Uso:
    python tools/visualize_step_response.py [--speed 3.0]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco_viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.characterize_system import (
    _clamp,
    build_parametrized_model,
    load_config,
    resolve_voltage_input,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=3.0,
                        help="Factor de velocidad respecto al tiempo real (default: 3.0)")
    parser.add_argument("--config", default="configs/sim_to_real_validation.toml")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_config(config_path)

    sim_cfg   = config["simulation"]
    input_cfg = config["input"]

    model_path = build_parametrized_model(config, ROOT)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)

    data.qpos[1] = math.pi
    mujoco.mj_forward(model, data)

    if "timestep_sec" in sim_cfg:
        model.opt.timestep = float(sim_cfg["timestep_sec"])

    duration    = float(sim_cfg["duration_sec"])
    max_voltage = float(config.get("motor", {}).get("max_voltage", 12.0))
    speed       = args.speed

    step_end = float(input_cfg.get("start_time_sec", 0.0)) + float(input_cfg["duration_sec"])

    print(f"Simulacion STEP_1023_100 — {duration:.0f} s a {speed:.1f}x velocidad real")
    print(f"Escalon: 0 — {step_end:.3f} s  |  Oscilacion libre: {step_end:.3f} — {duration:.0f} s")
    print("Cierra la ventana para terminar.\n")

    RENDER_FPS       = 30
    steps_per_render = max(1, round(speed / (RENDER_FPS * model.opt.timestep)))

    print(f"Rendering: {RENDER_FPS} FPS objetivo — {steps_per_render} pasos sim por frame")

    viewer     = mujoco_viewer.MujocoViewer(model, data)
    wall_start = time.time()

    while viewer.is_alive and data.time <= duration:
        for _ in range(steps_per_render):
            if data.time > duration:
                break
            t       = float(data.time)
            voltage = _clamp(resolve_voltage_input(t, input_cfg), -max_voltage, max_voltage)
            data.ctrl[0] = voltage
            mujoco.mj_step(model, data)

        viewer.render()

        target_wall = data.time / speed
        elapsed     = time.time() - wall_start
        slack       = target_wall - elapsed
        if slack > 0.001:
            time.sleep(slack)

    print("Simulacion terminada. Cerrando ventana...")
    viewer.close()


if __name__ == "__main__":
    main()
