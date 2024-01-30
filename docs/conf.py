import os
import sys

sys.path.insert(0, os.path.abspath("../"))
project = "FIFE"
copyright = "2018 - 2024, Institute for Defense Analyses"
author = "Institute for Defense Analyses"
release = "1.6.2"
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "m2r"]
html_theme = "sphinx_rtd_theme"
source_suffix = [".rst", ".md"]
autodoc_mock_imports = [
    "numpy",
    "dask",
    "tensorflow",
    "lifelines",
    "matplotlib",
    "pandas",
    "seaborn",
    "sklearn",
    "optuna",
    "shap",
    "lightgbm",
    "fife.survival_modeler",
]
