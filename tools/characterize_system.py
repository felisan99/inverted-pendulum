from __future__ import annotations

import argparse
import csv
import math
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy as media
import mujoco


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def _get_required_section(config: dict, section: str) -> dict:
    if section not in config:
        raise KeyError(f"Missing required section [{section}] in config file")
    return config[section]


def _set_joint_parameter(root: ET.Element, joint_name: str, parameter_name: str, value: float) -> None:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Joint '{joint_name}' was not found in XML model")
    joint.set(parameter_name, str(value))


def _set_general_actuator_param(root: ET.Element, actuator_name: str, param: str, value: str) -> None:
    actuator = root.find(f".//actuator/general[@name='{actuator_name}']")
    if actuator is None:
        raise ValueError(f"General actuator '{actuator_name}' was not found in XML model")
    actuator.set(param, value)


_FRICTION_DEFAULTS: dict = {
    "joint1_damping": 0.00198,
    "joint1_frictionloss": 0.02,
    "encoder_damping": 3.01e-4,
    "encoder_frictionloss": 0.0,
}

_MOTOR_DEFAULTS: dict = {
    "max_voltage": 12.0,
    "gainprm": 0.2184,
    "biasprm": -0.2385,
    "stall_torque_nm": 2.540,
    "resistance_ohm": 5.0,
    "torque_constant": 1.092,
    "back_emf_constant": 1.092,
}


def build_parametrized_model(config: dict, workspace_root: Path) -> Path:
    simulation_cfg = _get_required_section(config, "simulation")
    friction_cfg = {**_FRICTION_DEFAULTS, **config.get("friction", {})}
    motor_cfg = {**_MOTOR_DEFAULTS, **config.get("motor", {})}

    xml_model_relative = simulation_cfg.get("xml_model", "models/pendulum_high_quality.xml")
    xml_model_path = (workspace_root / xml_model_relative).resolve()
    if not xml_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_model_path}")

    tree = ET.parse(xml_model_path)
    root = tree.getroot()

    _set_joint_parameter(root, "joint1", "damping", float(friction_cfg.get("joint1_damping", 0.0)))
    _set_joint_parameter(root, "joint1", "frictionloss", float(friction_cfg.get("joint1_frictionloss", 0.05)))
    _set_joint_parameter(root, "encoder_joint", "damping", float(friction_cfg.get("encoder_damping", 0.0)))
    _set_joint_parameter(root, "encoder_joint", "frictionloss", float(friction_cfg.get("encoder_frictionloss", 0.0)))

    gainprm = float(motor_cfg["gainprm"])
    biasprm_vel = float(motor_cfg["biasprm"])
    stall_torque = float(motor_cfg.get("stall_torque_nm", 0.824))
    max_voltage = float(motor_cfg.get("max_voltage", 12.0))

    _set_general_actuator_param(root, "dc_motor", "gainprm", str(gainprm))
    _set_general_actuator_param(root, "dc_motor", "biasprm", f"0 0 {biasprm_vel}")
    _set_general_actuator_param(root, "dc_motor", "ctrlrange", f"{-max_voltage} {max_voltage}")
    _set_general_actuator_param(root, "dc_motor", "forcerange", f"{-stall_torque} {stall_torque}")

    tmp_file = tempfile.NamedTemporaryFile(prefix="pendulum_characterization_", suffix=".xml", delete=False)
    tmp_file.close()
    tree.write(tmp_file.name)
    return Path(tmp_file.name)


def step_voltage_signal(t: float, amplitude: float, start_time: float, duration: float) -> float:
    if start_time <= t < (start_time + duration):
        return amplitude
    return 0.0


def resolve_voltage_input(t: float, input_cfg: dict) -> float:
    input_type = input_cfg.get("type", "step").lower()

    if input_type != "step":
        raise ValueError(f"Unsupported input type '{input_type}'. Only 'step' is currently supported")

    return step_voltage_signal(
        t=t,
        amplitude=float(input_cfg["amplitude_voltage"]),
        start_time=float(input_cfg.get("start_time_sec", 0.0)),
        duration=float(input_cfg["duration_sec"]),
    )


def resolve_initial_pendulum_position(initial_cfg: dict | None) -> float:
    if not initial_cfg:
        return 0.0

    if "pendulum_position_rad" in initial_cfg:
        return float(initial_cfg["pendulum_position_rad"])

    position_label = str(initial_cfg.get("pendulum_position", "up")).lower()
    if position_label == "up":
        return 0.0
    if position_label == "down":
        return math.pi

    raise ValueError(
        "Invalid initial pendulum position. Use pendulum_position='up'|'down' or pendulum_position_rad"
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def estimate_motor_current(voltage: float, joint_velocity: float, motor_cfg: dict) -> float:
    """Estimate motor current for logging using the electrical model: I = (V - Kv*omega) / R."""
    kv = float(motor_cfg["back_emf_constant"])
    resistance = float(motor_cfg["resistance_ohm"])
    max_current_amps = float(motor_cfg.get("max_current_amps", 1e9))
    current = (voltage - kv * joint_velocity) / resistance
    return _clamp(current, -max_current_amps, max_current_amps)


def run_characterization(config: dict, workspace_root: Path) -> tuple[list[dict], list]:
    simulation_cfg = _get_required_section(config, "simulation")
    motor_cfg = {**_MOTOR_DEFAULTS, **config.get("motor", {})}
    input_cfg = _get_required_section(config, "input")
    output_cfg = _get_required_section(config, "output")

    model_path = build_parametrized_model(config, workspace_root)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    initial_cfg = config.get("initial_conditions", {})
    data.qpos[0] = float(initial_cfg.get("motor_position_rad", 0.0))
    data.qpos[1] = resolve_initial_pendulum_position(initial_cfg)
    mujoco.mj_forward(model, data)

    if "timestep_sec" in simulation_cfg:
        model.opt.timestep = float(simulation_cfg["timestep_sec"])

    duration_sec = float(simulation_cfg["duration_sec"])
    max_voltage = float(motor_cfg.get("max_voltage", 12.0))

    samples: list[dict] = []
    video_frames: list = []

    save_video = bool(output_cfg.get("save_video", False))
    video_fps = float(output_cfg.get("video_fps", 60.0))
    requested_video_width = int(output_cfg.get("video_width", 1280))
    requested_video_height = int(output_cfg.get("video_height", 720))

    renderer = None
    next_capture_time = 0.0
    if save_video:
        max_offscreen_width = int(model.vis.global_.offwidth)
        max_offscreen_height = int(model.vis.global_.offheight)
        video_width = min(requested_video_width, max_offscreen_width)
        video_height = min(requested_video_height, max_offscreen_height)

        if video_width <= 0 or video_height <= 0:
            raise ValueError("Invalid offscreen framebuffer size in model visual settings")

        if video_width != requested_video_width or video_height != requested_video_height:
            print(
                "Requested video size "
                f"{requested_video_width}x{requested_video_height} exceeds offscreen buffer "
                f"{max_offscreen_width}x{max_offscreen_height}. Using {video_width}x{video_height}."
            )

        renderer = mujoco.Renderer(model, width=video_width, height=video_height)
        renderer.update_scene(data)
        video_frames.append(renderer.render())
        next_capture_time = 1.0 / video_fps

    while data.time <= duration_sec:
        time_sec = float(data.time)

        joint_position = float(data.qpos[0])
        joint_velocity = float(data.qvel[0])

        voltage = _clamp(resolve_voltage_input(time_sec, input_cfg), -max_voltage, max_voltage)
        data.ctrl[0] = voltage

        motor_current = estimate_motor_current(voltage, joint_velocity, motor_cfg)
        motor_torque = float(motor_cfg["torque_constant"]) * motor_current
        motor_shaft_speed = 78.0 * joint_velocity

        samples.append(
            {
                "timestamp_sec": time_sec,
                "motor_position_rad": joint_position,
                "motor_speed_rad_s": joint_velocity,
                "motor_shaft_speed_rad_s": motor_shaft_speed,
                "motor_current_a": motor_current,
                "motor_torque_nm": motor_torque,
                "pendulum_position_rad": float(data.qpos[1]),
                "input_voltage": voltage,
            }
        )

        mujoco.mj_step(model, data)

        if renderer is not None:
            while data.time >= next_capture_time and next_capture_time <= duration_sec:
                renderer.update_scene(data)
                video_frames.append(renderer.render())
                next_capture_time += 1.0 / video_fps

    if renderer is not None:
        renderer.close()

    return samples, video_frames


def save_csv(samples: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_sec",
        "motor_position_rad",
        "motor_speed_rad_s",
        "motor_shaft_speed_rad_s",
        "motor_current_a",
        "motor_torque_nm",
        "pendulum_position_rad",
        "input_voltage",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def save_plot(samples: list[dict], plot_path: Path, show_plot: bool) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    time_values = [row["timestamp_sec"] for row in samples]
    motor_pos = [row["motor_position_rad"] for row in samples]
    motor_speed = [row["motor_speed_rad_s"] for row in samples]
    pendulum_pos = [row["pendulum_position_rad"] for row in samples]
    voltage_values = [row["input_voltage"] for row in samples]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(time_values, motor_pos, color="#1f77b4")
    axes[0].set_ylabel("Motor [rad]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_values, motor_speed, color="#9467bd")
    axes[1].set_ylabel("Motor vel [rad/s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_values, pendulum_pos, color="#2ca02c")
    axes[2].set_ylabel("Pendulum [rad]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_values, voltage_values, color="#d62728")
    axes[3].set_ylabel("Input [V]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)

    if show_plot:
        plt.show()

    plt.close(fig)


def save_video(frames: list, fps: float, video_path: Path) -> None:
    if not frames:
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(video_path), frames, fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pendulum simulation characterization from a TOML config")
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (workspace_root / config_path).resolve()

    config = load_config(config_path)
    samples, video_frames = run_characterization(config, workspace_root)

    output_cfg = _get_required_section(config, "output")
    csv_path = Path(output_cfg["csv_path"])
    plot_path = Path(output_cfg["plot_path"])
    show_plot = bool(output_cfg.get("show_plot", False))
    save_video_output = bool(output_cfg.get("save_video", False))
    video_fps = float(output_cfg.get("video_fps", 60.0))
    video_path_value = output_cfg.get("video_path", "data/step_12v_0p5s.mp4")
    video_path = Path(video_path_value)

    if not csv_path.is_absolute():
        csv_path = workspace_root / csv_path
    if not plot_path.is_absolute():
        plot_path = workspace_root / plot_path
    if not video_path.is_absolute():
        video_path = workspace_root / video_path

    save_csv(samples, csv_path)
    save_plot(samples, plot_path, show_plot)
    if save_video_output:
        save_video(video_frames, video_fps, video_path)

    print(f"Saved {len(samples)} samples to: {csv_path}")
    print(f"Saved plot to: {plot_path}")
    if save_video_output:
        print(f"Saved video to: {video_path}")


if __name__ == "__main__":
    main()
