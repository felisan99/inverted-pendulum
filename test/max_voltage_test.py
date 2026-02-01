"""
Script de prueba para validar la dinámica del motor DC.
Aplica voltaje máximo (12V) constante para verificar física y colisiones.

Uso:
    python test/max_voltage_test.py [--xml RUTA_AL_XML]
"""
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import mujoco
from gym_envs.pendulum_env import PendulumEnv

import argparse

def run_max_voltage_test(xml=None):

    print("Iniciando prueba de Máximo Voltaje (12V)")
    if xml:
        print(f"Usando el modelo XML: {xml}")
    
    env = PendulumEnv(model_path=xml, render_mode="human", max_steps=1000)
    
    obs, info = env.reset()
    
    try:
        while True:
            action = np.array([12.0], dtype=np.float32)
            
            obs, reward, done, truncated, info = env.step(action)
            
            # obs: [sin(th_m), cos(th_m), vel_m, sin(th_p), cos(th_p), vel_p]
            vel_motor = obs[2]
            vel_pendulo = obs[5]
            rpm = info.get("rpm_motor", 0.0)
            torque = info.get("torque", 0.0)
            
            if env.current_step % 50 == 0:
                print(f"Paso: {env.current_step} | Voltaje: 12V | RPM: {rpm:.1f} | Torque: {torque:.4f} Nm")

            if done:
                print("Episodio terminado.")
                obs, info = env.reset()
                time.sleep(1.0)
            
            if env.data.ncon > 0:
                print(f"Detectados {env.data.ncon} colisiones entre objetos del modelo XML.")

    except KeyboardInterrupt:
        print("\nPrueba detenida.")
    
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de máximo voltaje para el péndulo")
    parser.add_argument("--xml", type=str, default="mujoco_sim/xml_models/pendulum_model_v2.xml", help="Ruta al archivo XML del modelo")
    args = parser.parse_args()
    
    run_max_voltage_test(xml=args.xml)