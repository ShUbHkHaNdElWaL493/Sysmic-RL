import argparse
import mujoco.viewer
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mjcf",
        type=str,
        default=os.path.join('mjcf', 'sysmic_v2_scene.xml'),
        help="Path to the MJCF file (.xml)"
    )
    args = parser.parse_args()
    if not os.path.exists(args.mjcf):
        raise FileNotFoundError(f'MJCF file not found: {args.mjcf}')
    mujoco.viewer.launch_from_path(str(args.mjcf))

if __name__ == '__main__':
    main()