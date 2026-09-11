import argparse
from calibra_drive.recalibration import TemperatureScaling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', type=str, default='outputs/')
    args = parser.parse_args()
    print("Running recalibration...")

if __name__ == "__main__":
    main()
