"""
Script de prueba para validar la dinámica del motor DC.
Aplica acciones aleatorias al entorno del péndulo.

Uso:
    python test/random_episode_test.py [--xml RUTA_AL_XML]
"""

import time
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym_envs.pendulum_env as pendulum_env

def run_random_agent(xml=None):

    env = pendulum_env.PendulumEnv(model_path=xml, render_mode="human", max_steps=10000)

    try:
        while True:
            action = env.action_space.sample()
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

    except KeyboardInterrupt:
        print("\nPrueba detenida.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplica acciones aleatorias al entorno del péndulo.")
    parser.add_argument("--xml", type=str, default="mujoco_sim/xml_models/pendulum_model_v2.xml", help="Ruta al archivo XML")
    args = parser.parse_args()
    
    run_random_agent(xml=args.xml)