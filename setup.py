#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='sea_norm',
      version='1.0.0',
      description='Time normalized superposed epoch analysis (1D and 2D)',
      author='Sam Walton, Kyle Murphy',
      author_email='samuel.walton.18@ucl.ac.uk',
      license='MIT License',
      license_file = 'LICENSE.md',
      url='https://github.com/samwalton7645/SEA_Code',
      install_requires=['python>=3.6, pandas>=1.1.5','numpy>=1.21.6','scipy>=1.2.1','tqdm>=4.36.1'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: MIT License',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Atmospheric Science',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.7',
                   ],
     )
