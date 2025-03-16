"""
Setup script for tiny_distill package.
"""
from setuptools import setup, find_packages

setup(
    name="tiny_distill",
    version="0.1.0",
    description="Ultra-low memory knowledge distillation for language models",
    author="AI Engineer",
    author_email="ai@example.com",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.26.0",
        "datasets>=2.8.0",
        "peft>=0.3.0",
        "bitsandbytes>=0.37.0",
        "accelerate>=0.16.0",
        "h5py>=3.7.0",
        "tqdm",
        "psutil",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "optional": [
            "xformers",
            "triton",
        ],
    },
    entry_points={
        "console_scripts": [
            "tiny-distill=tiny_distill.main:main",
            "tiny-distill-profile=tiny_distill.scripts.memory_profile:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)