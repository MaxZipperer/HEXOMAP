from setuptools import setup, find_packages
import os
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('examples/johnson_aug18_demo')
setup(name='hexomap',
      version='0.3',
      packages=find_packages(),
      package_data={'hexomap': ['data/fundamental_zone/*','data/materials/*','kernel_cuda/*']+extra_files},
      scripts=['scripts/recon_mpi.py','scripts/recon_multigpu.py','scripts/reduction.py','scripts/recon.py'],
      )