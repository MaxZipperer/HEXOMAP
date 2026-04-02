# HEXOMAP: High Energy X-ray Orientation Mapping

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sry8vfFX_a9gJc084XpH12ZHFM4BPNb7#forceEdit=true&offline=true&sandboxMode=true)

[//]: # (https://colab.research.google.com/drive/1I5FUynlmLbwlF1nrRSE7bUSrRi6GVGGS#sandboxMode=true)

__HEXOMAP__ is a Cuda-based (realized through pycuda) near-filed high-energy
X-ray diffraction ([NF-HEDM](https://www.andrew.cmu.edu/user/suter/3dxdm/3dxdm.html))
reconstruction toolkit that provides 3D microstructure reconstructed with high
fadelity and efficiency.

> NOTE:  
> This GPU-based reconstruction toolkit is currently under development, and
> the API is subjected to change in the final stable release.

## Installation (written for Ubuntu 24.04.4)
It is assumed that you have installed conda. Instructions for this can be found at (https://www.anaconda.com/docs/getting-started/miniconda/install/overview)


1. Pull HEXOMAP from Git
   * Navigate to the directory you want HEXOMAP to live in
   * ```git clone https://github.com/MaxZipperer/HEXOMAP.git```
   * ```cd HEXOMAP```
3. Create the Conda Environment
   * ```conda env create --file hexomap_env.yml --name env_name```
   * Change env_name to whatever you like
   * ```conda activate env_name```
4. Install HEXOMAP
   * ```python setup.py install```
5. Verify
   * At the moment (at least for me) pycuda ignores local installation in favor of global installs
   * ```export PYCUDA_DEFAULT_NVCC_FLAGS="-allow-unsupported-compiler -ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"```
   * ```python -m hexomap```
   * You'll have to rerun the export step everytime after the conda activate step

## Usage and Examples
1. reconstruction	
    * see jupyter notebook: demonotebooks/*, it contains a full recontruction step ( parameter optimization and recosntruction).
    *    recon.py --config config.yml
    *    mpirun -n 4 recon_mpi.py --config config.yml
1. reduction
    *    mpirun -n 6 reduction.py

## Roadmap
* Support for lower symmetries: tetragonal, orthorhombic, trigonal, monoclinic, triclinic
* Multiple median reconstruction
* Voxelized strain state reconstruction

## License
__BSD 3 Cluase Licence__

## Notice
