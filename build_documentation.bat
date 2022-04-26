rem | This batch file generates documentation for FIFE
rem | This file will not automatically update the list of modules
rem | To update the list of modules, edit docs/source/fife.rst
rem | You may also wish to edit README.md and the files in docs/
rem | The documentation front page will be docs/_build/index.html
rem | This file requires an Anaconda (or Miniconda) installation
rem | You may need to specify your local path to activate.bat

call C:/Users/%username%/Miniconda3/Scripts/activate.bat
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat
call conda create -y -n fife_doc_env python=3.8
call C:/Users/%username%/Miniconda3/Scripts/activate.bat fife_doc_env
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat fife_doc_env
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade sphinx==2.4.4 sphinx_rtd_theme m2r jinja2==3.0.3 mistune==0.8.4 
cd docs/
sphinx-build . _build/
cmd /k
