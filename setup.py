from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
import os

PACKAGENAME='twopoppy'
VERSION='0.2.0'

# define custom build class

class mybuild(_build_py):
    def run(self):
        if not self.dry_run:
            
            print('Making version file')
            
            target_dir = os.path.join(self.build_lib, PACKAGENAME)
            
            # mkpath is a distutils helper to create directories
            self.mkpath(target_dir)

            with open(os.path.join(target_dir, '_version.py'), 'w') as f:
                f.write('__version__ = \'{}\''.format(VERSION))

        # continue normally

        _build_py.run(self)


setup(name=PACKAGENAME,
    version=VERSION,
    description='two-population dust evolution model according to Birnstiel, Klahr, Ercolano, A&A (2012)',
    long_descrioption=open(os.path.join(os.path.dirname(__file__),'README.md')).read(),
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
    entry_points={
        'console_scripts': [
            'twopoppy=twopoppy:main',
        ]
    },
    zip_safe=False,
    cmdclass={'build_py':mybuild}
    )
    
