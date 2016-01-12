# TWO-POP-PY                                                                  

This python script runs a two-population dust evolution model according to [Birnstiel,
Klahr, Ercolano, A&A (2012)](http://dx.doi.org/10.1051/0004-6361/201118136). Parameters
can be set by using the arguments, otherwise default parameters are used (see output). The
parameters, their meaning and units can be seen by executing `./two-pop-py.py -h`.

This code is published on [github.com/birnstiel](https://github.com/birnstiel/two-pop-py).

For bug reports, questions, ... contact me via [my website](http://www.til-birnstiel.de/contact/).

### Output Description

Output is written in the folder `data/` by default (can be specified with option `-dir`).
The following files are created:

|File	| Description	| Units  
|-------------	| ---	| ---	|  
|`x.dat`	| Radial grid	| cm  
|`T.dat`	| Temperature	| K  
|`a.dat`	| Grain size grid	| cm  
|`a_df.dat`	| drift-fragmentation limit on radial grid	| cm  
|`a_dr.dat`	| drift size limit on radial grid	| cm  
|`a_fr.dat`	| fragmentation limit on radial grid	| cm  
|`a_t.dat`	| maximum particle size as function of radius and time	| cm  
|`constants.dat`	| lists several constants	| see file contents  
|`sigma_d.dat`	| dust surface density as function of radius and time	| g cm^-2  
|`sigma_d_a.dat`	| final dust surface density distribution (fct. of particle size and radius)	| g cm^-2  
|`sigma_g.dat`	| gas surface density as function of radius and time	| g cm^-2  
|`time.dat`	| times at which the snapshots were taken	| s  
|`v_0.dat`	| small grain velocity as function of radius and time	| cm s^-1  
|`v_1.dat`	| large grain velocity as function of radius and time	| cm s^-1  
|`v_gas.dat`	| gas velocity as function of radius and time	| cm s^-1  

### Package dependencies

`astropy`, `numpy`, `scipy`

### Upcoming features:

- [ ] proper integration of $da/dt$ instead of using exponential approximation.


-------------
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) 