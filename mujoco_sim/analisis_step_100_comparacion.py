#!/usr/bin/env python3
"""
analisis_step_100_comparacion.py

Replica el experimento STEP_1023_100 en MuJoCo y compara con los datos reales.
Aplica el mismo metodo de decremento logaritmico que analisis_pendulo_libre.py.

Uso con config explicita:
    python mujoco_sim/analisis_step_100_comparacion.py \
        --config configs/sim_to_real_validation.toml \
        --real   <ruta_al_CSV_real>

Uso sin config (parametros y senal extraidos del CSV real):
    python mujoco_sim/analisis_step_100_comparacion.py \
        --real <ruta_al_CSV_real>

Opciones de salida:
    --output <ruta.png>   Guarda la grafica en la ruta indicada.
    --output              Omite la generacion de la grafica (modo headless).
    (sin --output)        Guarda en results/characterization/step_1023_100_comparacion.png
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mujoco_sim.characterize_system import load_config, run_characterization


AS5600_COUNTS  = 4096
MOTOR_COUNTS   = 1716
FS_REAL        = 1000.0  # Hz — tasa de muestreo del firmware

GAMMA = 1.8252e-3
DELTA = 7.168e-2
OMEGA_N_MODELO = math.sqrt(DELTA / GAMMA)

_DEFAULT_CONFIG = {
    "simulation": {
        "xml_model": "mujoco_sim/xml_models/pendulum_model_v3.xml",
        "timestep_sec": 0.001,
        "duration_sec": 12.0,
    },
    "initial_conditions": {
        "motor_position_rad": 0.0,
        "pendulum_position": "down",
    },
    "output": {
        "csv_path": "results/characterization/step_1023_100_sim.csv",
        "plot_path": "results/characterization/step_1023_100_sim.png",
        "show_plot": False,
        "save_video": False,
    },
}


def extract_step_params_from_csv(pwm, t):
    step_on = np.where(pwm > 0)[0]
    if len(step_on) == 0:
        raise ValueError("No se encontro senal de step en la columna PWM del CSV.")
    amplitude_voltage = float(np.max(pwm)) * 12.0 / 1023
    start_time_sec = float(t[step_on[0]])
    duration_sec = float(t[step_on[-1]]) - start_time_sec
    return amplitude_voltage, start_time_sec, duration_sec


def load_real_csv(csv_path: Path):
    rows = []
    with csv_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("t_us"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    data = np.array(rows)
    t       = data[:, 0] / 1e6
    pend_enc = data[:, 2]
    pwm      = data[:, 3]
    phi = pend_enc * (2 * math.pi / AS5600_COUNTS)
    phi = phi - phi[0]
    return t, phi, pwm


def extract_free_oscillation(t, phi, pwm):
    step_on    = np.where(pwm > 0)[0]
    t_step_end = t[step_on[-1]]
    idx_free   = np.where(t > t_step_end)[0]
    return t[idx_free], phi[idx_free], t_step_end


def extract_free_oscillation_sim(t_sim, phi_sim, step_duration=0.1):
    idx_free = np.where(t_sim > step_duration)[0]
    return t_sim[idx_free], phi_sim[idx_free]


def detect_peaks_and_fit(t_free, phi_free, fs):
    min_dist = int(0.4 * fs)
    pos_peaks, _ = find_peaks(phi_free, distance=min_dist, prominence=0.03)
    if len(pos_peaks) < 3:
        pos_peaks, _ = find_peaks(phi_free, distance=min_dist, prominence=0.01)
    N = len(pos_peaks)
    if N < 3:
        return None

    peak_t   = t_free[pos_peaks]
    peak_amp = phi_free[pos_peaks]

    Td      = float(np.mean(np.diff(peak_t)))
    omega_d = 2 * math.pi / Td
    delta_log = math.log(peak_amp[0] / peak_amp[-1]) / (N - 1)
    zeta      = delta_log / math.sqrt(4 * math.pi**2 + delta_log**2)
    omega_n   = omega_d / math.sqrt(1 - zeta**2)
    b2        = 2 * zeta * math.sqrt(GAMMA * DELTA)

    def damped_sin(t_rel, A, z, wn, phi0):
        wd = wn * math.sqrt(1 - z**2)
        return A * np.exp(-z * wn * t_rel) * np.cos(wd * t_rel + phi0)

    t_rel = t_free - t_free[0]
    fit_result = None
    try:
        p0 = [peak_amp[0], 0.012, 6.26, 0.0]
        lb = [-0.6, 0.001,  4.0, -math.pi]
        ub = [ 0.6, 0.20,  10.0,  math.pi]
        popt, _ = curve_fit(damped_sin, t_rel, phi_free, p0=p0, bounds=(lb, ub), maxfev=30000)
        fit_result = popt
    except Exception:
        pass

    return {
        "N":        N,
        "peak_t":   peak_t,
        "peak_amp": peak_amp,
        "Td":       Td,
        "omega_d":  omega_d,
        "zeta":     zeta,
        "omega_n":  omega_n,
        "b2":       b2,
        "fit":      fit_result,
        "damped_sin_fn": damped_sin,
        "t_free":   t_free,
        "phi_free": phi_free,
    }


def print_comparison(real_res, sim_res):
    SEP = "=" * 62
    print(f"\n{SEP}")
    print(f"  {'Parametro':<30} {'Real':>10}  {'Simulacion':>10}  {'Error %':>8}")
    print(SEP)
    params = [
        ("Picos detectados",    "N",       "",     False),
        ("Td [s]",              "Td",      "s",    True),
        ("omega_d [rad/s]",     "omega_d", "rad/s",True),
        ("zeta [-]",            "zeta",    "",     True),
        ("omega_n [rad/s]",     "omega_n", "rad/s",True),
        ("b2 [N·m·s/rad]",      "b2",      "",     True),
    ]
    for label, key, _, do_err in params:
        rv = real_res[key]
        sv = sim_res[key]
        if do_err and isinstance(rv, float) and rv != 0:
            err = abs(sv - rv) / rv * 100
            print(f"  {label:<30} {rv:>10.4f}  {sv:>10.4f}  {err:>7.2f}%")
        else:
            print(f"  {label:<30} {rv:>10}  {sv:>10}")
    print(f"\n  Referencia modelo: omega_n = {OMEGA_N_MODELO:.4f} rad/s")
    print(SEP)


def plot_comparison(t_real, phi_real, pwm_real, t_step_end_real,
                    t_sim_full, phi_sim_full,
                    real_res, sim_res,
                    out_path: Path | None):

    if out_path is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle(
        "Comparacion STEP_1023_100 — Real vs Simulacion (v3)\n"
        f"$\\omega_n$: real={real_res['omega_n']:.4f} rad/s  "
        f"sim={sim_res['omega_n']:.4f} rad/s  "
        f"modelo={OMEGA_N_MODELO:.4f} rad/s  |  "
        f"$\\zeta$: real={real_res['zeta']:.4f}  sim={sim_res['zeta']:.4f}",
        fontsize=10
    )

    # Panel 1: senal completa
    ax = axes[0]
    step_on = np.where(pwm_real > 0)[0]
    ax.plot(t_real, np.rad2deg(phi_real), color='darkorange', lw=0.8, label='Real (AS5600)', zorder=2)
    ax.plot(t_sim_full, np.rad2deg(phi_sim_full), color='steelblue', lw=0.8,
            label='Simulacion (v3)', alpha=0.85, zorder=1)
    ax.axvspan(t_real[step_on[0]], t_step_end_real, alpha=0.12, color='gray', label='Escalon 100 ms')
    ax.axvline(t_step_end_real, color='gray', ls='--', lw=0.8)
    ax.set_ylabel('$\\varphi$ [°]')
    ax.set_xlabel('Tiempo [s]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Posicion del pendulo — senal completa', fontsize=9)

    # Panel 2: oscilacion libre con picos y envolvente
    ax = axes[1]
    r = real_res
    s = sim_res

    # Tiempos relativos al fin del escalon (t=0 = momento en que cae el PWM)
    tr = r["t_free"] - r["t_free"][0]
    ts = s["t_free"] - s["t_free"][0]

    ax.plot(tr, np.rad2deg(r["phi_free"]),
            color='darkorange', lw=0.8, label='Real', zorder=2)
    ax.plot(ts, np.rad2deg(s["phi_free"]),
            color='steelblue', lw=0.8, label='Simulacion', alpha=0.85, zorder=1)

    r_peak_t_rel = r["peak_t"] - r["t_free"][0]
    s_peak_t_rel = s["peak_t"] - s["t_free"][0]

    ax.plot(r_peak_t_rel, np.rad2deg(r["peak_amp"]),
            'rv', ms=6, zorder=5, label=f'Picos real ({r["N"]})')
    ax.plot(s_peak_t_rel, np.rad2deg(s["peak_amp"]),
            'b^', ms=5, zorder=5, label=f'Picos sim ({s["N"]})', alpha=0.8)

    t_env = np.linspace(r_peak_t_rel[0], r_peak_t_rel[-1], 800)
    env_r = r["peak_amp"][0] * np.exp(-r["zeta"] * r["omega_n"] * (t_env - r_peak_t_rel[0]))
    ax.plot(t_env, np.rad2deg(env_r),  'k--', lw=1.0, alpha=0.6, label='Envolvente real')
    ax.plot(t_env, np.rad2deg(-env_r), 'k--', lw=1.0, alpha=0.6)

    t_env_s = np.linspace(s_peak_t_rel[0], s_peak_t_rel[-1], 800)
    env_s = s["peak_amp"][0] * np.exp(-s["zeta"] * s["omega_n"] * (t_env_s - s_peak_t_rel[0]))
    ax.plot(t_env_s, np.rad2deg(env_s),  'b--', lw=1.0, alpha=0.5)
    ax.plot(t_env_s, np.rad2deg(-env_s), 'b--', lw=1.0, alpha=0.5)

    ax.set_ylabel('$\\varphi$ [°]')
    ax.set_xlabel('Tiempo desde fin del escalon [s]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Oscilacion libre post-escalon (tiempo desde fin del escalon)', fontsize=9)

    # Panel 3: decaimiento log en escala logaritmica
    ax = axes[2]
    ax.semilogy(r["peak_t"] - r["peak_t"][0], r["peak_amp"],
                'rs', ms=6, label=f'Real: $\\omega_n$={r["omega_n"]:.4f}, $\\zeta$={r["zeta"]:.4f}')
    ax.semilogy(s["peak_t"] - s["peak_t"][0], s["peak_amp"],
                'b^', ms=5, alpha=0.8,
                label=f'Sim: $\\omega_n$={s["omega_n"]:.4f}, $\\zeta$={s["zeta"]:.4f}')

    t_env_log = np.linspace(0, max(r["peak_t"][-1] - r["peak_t"][0],
                                   s["peak_t"][-1] - s["peak_t"][0]), 800)
    ax.semilogy(t_env_log,
                r["peak_amp"][0] * np.exp(-r["zeta"] * r["omega_n"] * t_env_log),
                'r--', lw=1.2, alpha=0.7)
    ax.semilogy(t_env_log,
                s["peak_amp"][0] * np.exp(-s["zeta"] * s["omega_n"] * t_env_log),
                'b--', lw=1.2, alpha=0.7)

    ax.set_ylabel('Amplitud [rad] — escala log')
    ax.set_xlabel('Tiempo desde primer pico [s]')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title('Decaimiento logaritmico de amplitudes', fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nGrafica guardada: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default=None,
                        help="TOML config path (relativo a la raiz del proyecto). "
                             "Si no se pasa, se usan los parametros validados del modelo v3 "
                             "y la senal de entrada se extrae del CSV.")
    parser.add_argument("--real", required=True, help="Ruta al CSV real (STEP_1023_100_*.csv)")
    parser.add_argument("--output", nargs="?", const="", default=None,
                        help="Ruta donde guardar la grafica. "
                             "Sin valor: omite la grafica. "
                             "Sin --output: guarda en results/characterization/.")
    args = parser.parse_args()

    real_csv = Path(args.real).resolve()
    if not real_csv.exists():
        sys.exit(f"CSV real no encontrado: {real_csv}")

    print("Cargando datos reales...")
    t_real, phi_real, pwm_real = load_real_csv(real_csv)
    t_free_real, phi_free_real, t_step_end_real = extract_free_oscillation(t_real, phi_real, pwm_real)

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (ROOT / config_path).resolve()
        config = load_config(config_path)
    else:
        print("Sin --config: extrayendo parametros de entrada del CSV y usando defaults del modelo v3.")
        amplitude_voltage, start_time_sec, duration_sec = extract_step_params_from_csv(pwm_real, t_real)
        config = {
            **_DEFAULT_CONFIG,
            "input": {
                "type": "step",
                "amplitude_voltage": amplitude_voltage,
                "start_time_sec": start_time_sec,
                "duration_sec": duration_sec,
            },
        }
        print(f"  Amplitud extraida: {amplitude_voltage:.2f} V  |  "
              f"Inicio: {start_time_sec:.4f} s  |  Duracion: {duration_sec:.4f} s")

    print("Ejecutando simulacion...")
    samples, _ = run_characterization(config, ROOT)

    t_sim  = np.array([s["timestamp_sec"]        for s in samples])
    q1_sim = np.array([s["pendulum_position_rad"] for s in samples])
    phi_sim = q1_sim - math.pi

    step_duration = config["input"]["duration_sec"]
    t_free_sim, phi_free_sim = extract_free_oscillation_sim(t_sim, phi_sim, step_duration=step_duration)

    print("Analizando oscilacion libre — real...")
    real_res = detect_peaks_and_fit(t_free_real, phi_free_real, FS_REAL)
    if real_res is None:
        sys.exit("No se detectaron suficientes picos en los datos reales.")

    print("Analizando oscilacion libre — simulacion...")
    sim_res = detect_peaks_and_fit(t_free_sim, phi_free_sim, 1.0 / config["simulation"]["timestep_sec"])
    if sim_res is None:
        sys.exit("No se detectaron suficientes picos en la simulacion.")

    print_comparison(real_res, sim_res)

    if args.output is None:
        out_path = ROOT / "results" / "characterization" / "step_1023_100_comparacion.png"
    elif args.output == "":
        out_path = None
    else:
        out_path = Path(args.output)

    plot_comparison(
        t_real, phi_real, pwm_real, t_step_end_real,
        t_sim, phi_sim,
        real_res, sim_res,
        out_path
    )


if __name__ == "__main__":
    main()
