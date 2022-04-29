"""Setup for FIFE: Finite-Interval Forecasting Engine."""

from setuptools import setup

with open("README.md", "r") as f:
    README = f.read()

setup(
    name="fife",
    version="1.6.0",
    description=(
        "Finite-Interval Forecasting Engine: Machine learning models "
        "for discrete-time survival analysis and multivariate time series "
        "forecasting"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/IDA-HumanCapital/fife",
    project_urls={
        "Bug Tracker": "https://github.com/IDA-HumanCapital/fife/issues",
        "Source Code": "https://github.com/IDA-HumanCapital/fife",
        "Documentation": "https://fife.readthedocs.io/en/latest",
    },
    author="Institute for Defense Analyses",
    author_email="humancapital@ida.org",
    license="AGPLv3+",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: "
        "GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=["fife"],
    install_requires=[
        "dask[delayed]",
        "ipython",
        "keras",
        "lifelines",
        "lightgbm",
        "matplotlib",
        "numpy",
        "optuna",
        "pandas",
        "seaborn",
        "scikit-learn",
    ],
    extras_require={"shap": ["shap"], "tensorflow": ["tensorflow"]},
    entry_points={
        "console_scripts": [
            "fife=fife.__main__:main",
        ]
    },
)
