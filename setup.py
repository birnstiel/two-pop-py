from setuptools import setup
import os

PACKAGENAME='twopoppy'
VERSION='0.2.2'

# define custom build class


with open(os.path.join(PACKAGENAME, '_version.py'), 'w') as f:
    f.write('__version__ = \'{}\''.format(VERSION))


setup(name=PACKAGENAME,
    version=VERSION,
    description='two-population dust evolution model according to Birnstiel, Klahr, Ercolano, A&A (2012)',
    long_description=open(os.path.join(os.path.dirname(__file__),'README.md')).read(),
    url='http://www.til-birnstiel.de',
    author='Til Birnstiel',
    author_email='birnstiel@me.com',
    license='GPLv3',
    packages=[PACKAGENAME],
    include_package_data=True,
    install_requires=[
        'astropy',
        'configobj',
        'scipy',
        'numpy',
        'matplotlib'
        ],

    scripts=['scripts/twopoppyrun'],
    zip_safe=False
    )
    
