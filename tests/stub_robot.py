"""Mesh-free stand-in for the Sysmic MJCF (same body/joint/actuator names and axes)
so the pipeline can be tested in CI without the STL meshes."""
from pathlib import Path


def _arm(side, y, ax):
    p = f"arm_{side}"
    pan, sh = ax["pan"], ax["sh"]
    J = lambda n, a, r: (f'<joint name="{p}_{n}_joint" axis="{a}" range="{r}" armature="0.01" damping="0.2"/>')
    cap = lambda l: f'<geom type="capsule" fromto="0 0 0 {l} 0 0" size="0.012" mass="0.2"/>'
    return f'''
    <body name="{p}_base_link" pos="0.112 {y} 0.1187">
      <inertial pos="0 0 0" mass="0.36" diaginertia="0.001 0.001 0.001"/>
      <body name="{p}_shoulder_pan_link">
        <inertial pos="0.03 0 0.04" mass="0.27" diaginertia="0.001 0.001 0.001"/>
        {J("pan", pan, "-1.91986 1.91986")}
        <body name="{p}_upper_arm_link" pos="0.031 0 0.07">
          <inertial pos="-0.06 0 0" mass="1.0" diaginertia="0.002 0.002 0.001"/>
          {J("shoulder", sh, "0 3.49066")}
          <geom type="capsule" fromto="0 0 0 -0.112 0 0" size="0.012" mass="0.1"/>
          <body name="{p}_forearm_link" pos="-0.112 0 0.024">
            <inertial pos="0.07 0 0" mass="0.9" diaginertia="0.003 0.003 0.001"/>
            {J("elbow", sh, "-3.38594 0")}
            {cap(0.134)}
            <body name="{p}_wrist_link" pos="0.134 0 0">
              <inertial pos="0.03 0 0" mass="0.33" diaginertia="0.0004 0.0004 0.0002"/>
              {J("wrist_flex", sh, "-1.65806 1.65806")}
              <body name="{p}_wrist_roll_link" pos="0.029 0 0">
                <inertial pos="0.058 0 0" mass="0.63" diaginertia="0.0009 0.0009 0.0006"/>
                {J("wrist_roll", "1 0 0", "-2.74017 2.84489")}
                <geom type="capsule" fromto="0 0 0 0.09 0 0" size="0.012" mass="0.1"/>
                <body name="{p}_jaw_link" pos="0.056 0 0.023">
                  <inertial pos="0 0.02 0.02" mass="0.02" diaginertia="2e-5 2e-5 1e-5"/>
                  {J("gripper", sh, "-0.174533 1.74533")}
                  <geom type="box" size="0.01 0.003 0.03" pos="0.02 0 0.03" mass="0.01"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>'''


def write_stub_robot(directory) -> str:
    left = _arm("left", 0.1146, {"pan": "0 0.5 0.866", "sh": "0 0.866 -0.5"})
    right = _arm("right", -0.1146, {"pan": "0 -0.5 0.866", "sh": "0 0.866 0.5"})
    pos = lambda n, lo, hi, j=None: f'<position name="{n}" joint="{j or n}" kp="200" kv="3" ctrlrange="{lo} {hi}" forcerange="-11 11"/>'
    acts = ['<position name="lift" joint="lift_joint" kp="200000" kv="10000" ctrlrange="0 0.447677" forcerange="-2000 2000"/>']
    for s in ("left", "right"):
        for n, lo, hi in [("pan", -1.919862, 1.919862), ("shoulder", 0, 3.490659), ("elbow", -3.385939, 0),
                          ("wrist_flex", -1.658063, 1.658063), ("wrist_roll", -2.740167, 2.844887),
                          ("gripper", -0.174533, 1.745329)]:
            acts.append(pos(f"arm_{s}_{n}_joint", lo, hi))
    xml = f'''<mujoco model="stub">
  <compiler angle="radian"/>
  <option integrator="implicitfast" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="sysmic_root"><body name="base_link">
      <inertial pos="0 0 0.3" mass="100" diaginertia="10 10 5"/>
      <geom type="box" size="0.06 0.2 0.3" pos="-0.17 0 0.3"/>
      <body name="lift_carriage_link" pos="-0.117 0 0.64">
        <inertial pos="0.07 0 0" mass="30" diaginertia="1 1 0.4"/>
        <joint name="lift_joint" type="slide" axis="0 0 1" range="0 0.447677"/>
        <geom type="box" size="0.02 0.2 0.05" pos="0.0 0 0.1" mass="1"/>
        <geom name="shelf" type="box" size="0.14 0.22 0.01" pos="0.20 0 0.03" mass="1"/>
        {left}
        {right}
      </body>
    </body></body>
    <body name="left_target" mocap="true"><geom type="sphere" size="0.03" contype="0" conaffinity="0"/></body>
    <body name="right_target" mocap="true"><geom type="sphere" size="0.03" contype="0" conaffinity="0"/></body>
  </worldbody>
  <contact><exclude body1="base_link" body2="lift_carriage_link"/></contact>
  <actuator>{"".join(acts)}</actuator>
</mujoco>'''
    p = Path(directory) / "stub_robot.xml"
    p.write_text(xml)
    return str(p)
