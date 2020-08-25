rem | This batch file generates documentation for FIFE
rem | This file will not automatically update the list of modules
rem | To update the list of modules, edit docs/source/fife.rst
rem | You may also wish to edit README.md and the files in docs/
rem | The documentation front page will be docs/_build/index.html
rem | This file requires an Anaconda (or Miniconda) installation
rem | You may need to specify your local path to activate.bat

call C:/Users/%username%/Miniconda3/Scripts/activate.bat
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade sphinx==2.4.4 sphinx_rtd_theme m2r
cd docs/
sphinx-build . _build/
cmd /k
