import mujoco
import mujoco.viewer
import time
import trayectorias

def simulation(plot: bool):
  #model = mujoco.MjModel.from_xml_path("my_model.xml")
  model = mujoco.MjModel.from_xml_path("pendulum_model.xml")
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







