rem | This batch file creates and tests a FIFE package
rem | You may wish to update the python and FIFE versions in this script
rem | This file requires an Anaconda (or Miniconda) installation
rem | You may need to specify your local path to activate.bat

@echo off
setlocal enabledelayedexpansion
call C:/Users/%username%/Miniconda3/Scripts/activate.bat
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat
call conda create -y -n fife_env python=3.8
call C:/Users/%username%/Miniconda3/Scripts/activate.bat fife_env
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat fife_env
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade setuptools wheel
rmdir build /s /q
python setup.py sdist bdist_wheel
call conda install -y -c conda-forge shap iniconfig
call conda install -y -c anaconda tensorflow
for /F %%i in ('python setup.py --version') do set version=%%i
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade dist/fife-%version%-py3-none-any.whl[shap] black pytest
black .
pytest
if '!errorlevel!' == '0' (
	python -c "from fife.utils import create_example_data; import pandas as pd; create_example_data().to_csv('Input_Data.csv', index=False)"
	fife example_config.json
	if '!errorlevel!' == '0' (
		fife --TEST_INTERVALS 4
		if '!errorlevel!' == '0' (
			fife --TREE_MODELS False
			if '!errorlevel!' == '0' (
				pip freeze > requirements.txt
				color 0A
				echo "Tests passed."
			) else (
				color 0c
				echo "Neural network functional test failed."
			)
		) else (
			color 0c
			echo "Evaluation functional test failed."
		)
	) else (
		color 0c
		echo "Forecasting functional test failed."
	)
) else (
	echo "Unit test failed."
)
cmd /k
