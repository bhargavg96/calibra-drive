import argparse
from calibra_drive.data import NuScenesLoader
from calibra_drive.models.occworld_wrapper import OccWorldWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/models/occworld.yaml')
    args = parser.parse_args()
    print("Running inference...")

if __name__ == "__main__":
    main()
