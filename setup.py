from setuptools import setup, find_packages

setup(
    name="calibra-drive",
    version="0.1.0",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "nuscenes-devkit",
        "pyquaternion",
        "tqdm",
        "pyyaml",
        "hydra-core"
    ],
)
