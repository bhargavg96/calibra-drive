import argparse
from calibra_drive.metrics import CalibrationMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', type=str, default='outputs/')
    args = parser.parse_args()
    print("Computing calibration...")

if __name__ == "__main__":
    main()
