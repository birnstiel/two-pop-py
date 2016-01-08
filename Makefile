all:
	-wget --no-check-certificate https://raw.githubusercontent.com/birnstiel/Birnstiel2015_scripts/master/distribution_reconstruction.py
	-wget --no-check-certificate https://raw.githubusercontent.com/birnstiel/Birnstiel2015_scripts/master/aux_functions.py

clean: 
	-rm aux_functions.py
	-rm aux_functions.pyc
	-rm const.pyc
	-rm distribution_reconstruction.py
	-rm distribution_reconstruction.pyc
	-rm two_pop_model.pyc
	
clobber: clean
	-rm -rf data
