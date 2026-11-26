import argparse
import mujoco.viewer
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mjcf",
        type=str,
        default=os.path.join('mjcf', 'sysmic_v2.xml'),
        help="Path to the MJCF file (.xml)"
    )
    args = parser.parse_args()
    if not os.path.exists(args.mjcf_path):
        raise FileNotFoundError(f'MJCF file not found: {args.mjcf_path}')
    mujoco.viewer.launch_from_path(str(args.mjcf_path))

if __name__ == '__main__':
    main()