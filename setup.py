"""
FATES-Emulator: Machine Learning Framework for Ecosystem Model Calibration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fates-emulator",
    version="0.1.0",
    author="Lingcheng Li",
    author_email="lingcheng.li@pnnl.gov",
    description="AutoML-based emulator framework for FATES ecosystem model calibration with coexistence constraints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pnnl/fates-emulator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9b0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fates-emulator=fates_emulator.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

