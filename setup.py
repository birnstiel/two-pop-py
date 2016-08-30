from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='twopoppy',
    version='0.1',
    description='two-population dust evolution model according to Birnstiel, Klahr, Ercolano, A&A (2012)',
    long_descrioption=read('README.md'),
    url='http://www.til-birnstiel.de',
    author='Til Birnstiel',
    author_email='birnstiel@me.com',
    license='GPLv3',
    packages=['twopoppy'],
    include_package_data=True,
    install_requires=[
        'astropy',
        'scipy',
        'numpy',
        'matplotlib'
        ],
    entry_points={
        'console_scripts': [
            'twopoppy=twopoppy:main',
        ]
    },
    zip_safe=False)
