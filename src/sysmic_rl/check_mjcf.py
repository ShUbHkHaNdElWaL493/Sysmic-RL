import mujoco.viewer
import os

def main() -> None:
    mjcf_path = os.path.join('mjcf', 'sysmic_v2.xml')
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f'MJCF file not found: {mjcf_path}')
    mujoco.viewer.launch_from_path(str(mjcf_path))

if __name__ == '__main__':
    main()