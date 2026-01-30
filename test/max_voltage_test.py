import numpy as np
import time
import sys
import os

# Ajustar path para encontrar gym_envs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import mujoco
from gym_envs.pendulum_env import PendulumEnv

def run_max_voltage_test():
    print("Iniciando prueba de Máximo Voltaje (12V)...")
    
    # Crear entorno con renderizado "human" para ver la ventana nativa
    # Importante: Ejecutar esto desde la terminal con python directamente, no desde notebooks
    env = PendulumEnv(render_mode="human", max_steps=5000)
    
    obs, info = env.reset()
    
    try:
        while True:
            # Acción constante de máximo voltaje positivo
            action = np.array([12.0], dtype=np.float32)
            
            obs, reward, done, truncated, info = env.step(action)
            
            # obs: [sin(th_m), cos(th_m), vel_m, sin(th_p), cos(th_p), vel_p]
            vel_motor = obs[2]
            vel_pendulo = obs[5]
            rpm = info.get("rpm_motor", 0.0)
            torque = info.get("torque", 0.0)
            
            # Imprimir telemetría básica cada 50 pasos para no saturar la terminal
            if env.current_step % 50 == 0:
                print(f"Paso: {env.current_step} | Voltaje: 12V | RPM: {rpm:.1f} | Torque: {torque:.4f} Nm")

            # Condición de salida
            if done:
                print("Episodio terminado.")
                obs, info = env.reset()
                time.sleep(1.0) # Pausa entre episodios
            
            # Debug de posibles colisiones
            if env.data.ncon > 0:
                print(f"ADVERTENCIA: Detectados {env.data.ncon} contactos (colisiones). Esto puede estar frenando el motor.")

    except KeyboardInterrupt:
        print("\nPrueba detenida por el usuario.")
    
    finally:
        env.close()

if __name__ == "__main__":
    run_max_voltage_test()
