import mujoco
import mujoco.viewer
import time
import mujoco_sim.trayectorias as trayectorias
from pathlib import Path

def simulation(plot: bool):
  ROOT_DIR = Path(__file__).resolve().parent.parent
  MODEL_PATH = ROOT_DIR / "models" / "pendulum_model.xml"

  model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
  data = mujoco.MjData(model)

  trajectory1 = []
  trajectory2 = []

  with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer is running...")
    t0 = time.time()

    while viewer.is_running():
      t_actual = t0 - time.time()
      mujoco.mj_step(model, data)

      current_position1 = data.qpos[0]
      current_position2 = data.qpos[1]

      trajectory1.append(current_position1)
      trajectory2.append(current_position2)

      data.qpos[0] = trayectorias.seno(t_actual, 2, 2)
          
      if (plot and (len(trajectory1) == 500)):
        break

      if not plot:
        trajectory1 = []
        trajectory2 = []

      viewer.sync()
      time.sleep(0.01)
      
  return trajectory1, trajectory2







