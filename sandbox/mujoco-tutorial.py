import time
import itertools
import numpy as np
import mujoco
import matplotlib.pyplot as plt


xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)

id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_sphere")
model.geom_rgba[id, :]

print('id of "green_sphere": ', model.geom("green_sphere").id)
print("name of geom 1: ", model.geom(1).name)
print("name of body 0: ", model.body(0).name)

data = mujoco.MjData(model)

print(data.geom_xpos)
mujoco.mj_kinematics(model, data)
print("raw access:\n", data.geom_xpos)

# MjData also supports named access:
print("\nnamed access:\n", data.geom("green_sphere").xpos)

data = mujoco.MjData(model)

# Make renderer, render and show the pixels
with mujoco.Renderer(model) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    plt.imshow(renderer.render(), interpolation="nearest", cmap="viridis")
    plt.show()
