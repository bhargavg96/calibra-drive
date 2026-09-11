import argparse
from calibra_drive.planning import UncertaintyAwarePlanner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/experiments/planning.yaml')
    args = parser.parse_args()
    print("Running planning...")

if __name__ == "__main__":
    main()
