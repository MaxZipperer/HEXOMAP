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

## Installation

### Python 3.13+ and CUDA 12+ (recommended)

1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/MaxZipperer/HEXOMAP.git
   cd HEXOMAP
   ```
2. Create a conda environment with Python 3.13 and CUDA 13:
   ```bash
   conda env create -f hexomap_env_py313.yml -n hexomap
   conda activate hexomap
   ```
3. Install HEXOMAP:
   ```bash
   pip install -e .
   ```
4. If NVCC rejects your compiler, set PyCUDA flags (Linux example):
   ```bash
   conda env config vars set PYCUDA_DEFAULT_NVCC_FLAGS="-allow-unsupported-compiler -ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
   conda deactivate && conda activate hexomap
   ```
5. Verify:
   ```bash
   python -m hexomap
   ```
6. Jupyter
   ```bash
   conda install -c conda-forge jupyter
   conda install ipykernel
   python -m ipykernel install --user --name=hexomap
   ```
   
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
