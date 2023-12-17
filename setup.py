# global imports
from setuptools import setup, find_packages

# library specifications
setup(name='dlearn',
      version='0.0.0',
      packages=find_packages(exclude=['examples', 'dlearn_tests']),
      install_requires=[
          'torch==2.1.1'
        ]
      )