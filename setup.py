from setuptools import setup, find_packages

setup(
    name="paft",
    version="0.1.0",
    description="Polar decomposition Attention Fine-Tuning — geometric PEFT for transformers",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "datasets>=2.18.0",
        "peft>=0.9.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "evaluate>=0.4.1",
        "scikit-learn>=1.4.0",
        "rouge-score>=0.1.2",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "analysis": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "tensorboard>=2.16.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
