all:
	-if [[ ! -f distribution_reconstruction.py ]]; then wget --no-check-certificate https://raw.githubusercontent.com/birnstiel/Birnstiel2015_scripts/master/distribution_reconstruction.py; fi
	-if [[ ! -f aux_functions.py ]]; then wget --no-check-certificate https://raw.githubusercontent.com/birnstiel/Birnstiel2015_scripts/master/aux_functions.py; fi

clean: 
	-rm aux_functions.py
	-rm aux_functions.pyc
	-rm const.pyc
	-rm distribution_reconstruction.py
	-rm distribution_reconstruction.pyc
	-rm two_pop_model.pyc
	
clobber: clean
	-rm -rf data
	-rm -rf *.pyc
	-rm -rf twopoppy.egg-info
	-rm -rf twopoppy/*.pyc
