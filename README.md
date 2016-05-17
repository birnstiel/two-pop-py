# TWO-POP-PY                                                                  

This python script runs a two-population dust evolution model according to [Birnstiel, Klahr, Ercolano, A&A (2012)](http://dx.doi.org/10.1051/0004-6361/201118136). Parameters can be set by using the arguments, otherwise default parameters are used (see output). The parameters, their meaning and units can be seen by executing `./two-pop-py.py -h`.

This code is published on [github.com/birnstiel](https://github.com/birnstiel/two-pop-py).

For bug reports, questions, ... contact me via [my website](http://www.til-birnstiel.de/contact/).

*If you use this code in a publication, please cite at least [Birnstiel, Klahr, Ercolano, A&A (2012)](http://dx.doi.org/10.1051/0004-6361/201118136), and possibly [Birnstiel et al. (ApJL) 2015](http://dx.doi.org/10.1088/2041-8205/813/1/L14) if you use the size distribution reconstruction. I addition to that, it would be best practice to include the hash of the version you used to make sure results are reproducible, as the code can change.*

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
[![DOI](https://zenodo.org/badge/0516/birnstiel/two-pop-py.svg)](https://zenodo.org/badge/latestdoi/0516/birnstiel/two-pop-py)
[![GPLv3](http://img.shields.io/badge/license-GPL-brightgreen.svg?style=flat)](https://github.com/birnstiel/two-pop-py/blob/master/LICENSE) [![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)  [![arxiv](https://img.shields.io/badge/arxiv-1510.05660-green.svg?style=flat)](http://arxiv.org/abs/1510.05660) [![arxiv](https://img.shields.io/badge/arxiv-1201.5781-green.svg?style=flat)](http://arxiv.org/abs/1201.5781) [![arxiv](https://img.shields.io/badge/arxiv-1009.3011-green.svg?style=flat)](http://arxiv.org/abs/1009.3011)