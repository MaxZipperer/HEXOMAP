# HEXOMAP v0.3 — Modernization Changelog

This document describes the code and environment updates made to run **HEXOMAP** on
**Python 3.13+** and **CUDA 12/13**, while keeping the existing package layout and
reconstruction workflow intact.

Branch: `v0.3`  
Baseline: `v0.2` (Python 3.9, CUDA 11.5, legacy CUDA texture references)

---

## Summary

HEXOMAP originally targeted Python 3.6–3.9 and CUDA 9–11. CUDA 12 removed several
legacy GPU APIs (notably **texture references**), and Python 3.13/3.14 introduced
stricter dataclass and packaging rules. Version 0.3 updates the GPU kernels, PyCUDA
integration, dependency pins, and a handful of Python compatibility fixes so the
demo reconstruction (`python -m hexomap`) runs on modern toolchains.

**No project restructuring was performed.** Scripts, config formats, and the overall
reconstruction API are unchanged.

---

## Environment Requirements

| Component | v0.2 (legacy) | v0.3 (recommended) |
|-----------|---------------|---------------------|
| Python    | 3.9           | 3.13+ (tested on 3.14) |
| CUDA      | 11.5          | 12.x / 13.x |
| PyCUDA    | older pins    | `>= 2024.1` |
| NumPy     | older pins    | `>= 1.26` |
| SciPy     | older pins    | `>= 1.11` |

### New conda environment file

`hexomap_env_py313.yml` provides a conda-forge + nvidia channel environment with
Python 3.13, CUDA toolkit 13, and pip-installed PyCUDA.

### Install (recommended)

```bash
git clone https://github.com/MaxZipperer/HEXOMAP.git
cd HEXOMAP
git checkout v0.3

conda env create -f hexomap_env_py313.yml -n hexomap
conda activate hexomap
pip install -e .
```

If NVCC rejects the system compiler on Linux clusters:

```bash
conda env config vars set PYCUDA_DEFAULT_NVCC_FLAGS="-allow-unsupported-compiler -ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
conda deactivate && conda activate hexomap
```

### Verify

```bash
python -m hexomap
```

This loads the Johnson Aug 18 gold demo config and runs a short serial reconstruction.

---

## CUDA / GPU Changes

### 1. Removed deprecated texture references

**Problem:** CUDA 12+ no longer supports the legacy `texture<>` reference API and
`tex2D` / `tex3D` intrinsics that the original kernels relied on.

**Solution:** Kernels now read directly from **device memory pointers** passed as
kernel arguments.

#### `hexomap/kernel_cuda/device_code.cu`

| Kernel | Before | After |
|--------|--------|-------|
| `simulation` | Read reciprocal lattice vectors `afG` via a bound 1D texture | `const float* __restrict__ afG` — indexed as `afG[threadIdx.x*3+j]` |
| `hitratio_multi_detector` | Read experimental peak map via a bound 3D texture (`tex3D`) | `const unsigned char* __restrict__ acExpData` with flat indexing `acExpData[iz*iExpNK*iExpNJ + iy*iExpNJ + ix]` |

#### `hexomap/reconstruction.py`

Corresponding host-side changes:

- `set_Q()` uploads `self.sample.Gs` to `self.afGD` via `gpuarray.to_gpu()` instead of binding a texture.
- `__cp_expdata_to_gpu()` uploads the CPU peak map to `self.acExpDataD` and stores `self.iExpNK`, `self.iExpNJ` for kernel indexing.
- All `CudaTextureObject` / texture-handle plumbing was removed from the reconstruction path.

An intermediate `hexomap/cuda_texture.py` helper (ctypes-based bindless texture objects) was explored but abandoned due to instability; it is **not used** by the current reconstruction code path.

### 2. Replaced `gpuarray.take()` (PyCUDA JIT kernels)

**Problem:** During reconstruction, `single_voxel_recon` called `gpuarray.take()` to
select orientation matrices. PyCUDA JIT-compiles elementwise kernels for `take()`, and
those kernels include `pycuda/cuda/pycuda-helpers.hpp`, which pulls in
`surface_functions.h` — a header **removed in CUDA 12**. This produces:

```
fatal error: surface_functions.h: No such file or directory
```

This is **not** a missing demo data file. Experimental data loads successfully before
this error appears.

**Solution:** Added `_gpu_select_orientations()` in `reconstruction.py`, which:

1. Copies the orientation matrix GPU array to host memory.
2. Selects rows by index (`reshape(-1, 9)[row_indices]`).
3. Uploads the result back to the GPU.

Both call sites in `single_voxel_recon` (twiddle and non-twiddle branches) now use this
helper instead of `gpuarray.take()`.

**Trade-off:** Selection happens on the CPU with a GPU↔host copy. This is acceptable
because it runs once per iteration on a small number of orientations (`NSelect`), not
on the full simulation grid.

### 3. Lazy CUDA kernel compilation in `cuorientations.py`

The `misorien` CUDA kernel is now compiled on first use via `_ensure_kernel()` rather
than at import time. This avoids NVCC compilation failures when importing the module
in environments without a visible GPU.

---

## Python 3.13 / 3.14 Compatibility

### Dataclass mutable defaults (`orientation.py`, `virtualdiffractor.py`)

Python 3.14 enforces dataclass rules against mutable `np.ndarray` defaults.
`Frame` and `Detector.frame` now use `field(default_factory=...)`:

```python
e1: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))
```

The same fix was applied to `orientation_legacy.py` for consistency.

### HDF5 byte strings (`utility.py`)

HDF5 datasets often return byte strings (e.g. `b'gold'` instead of `'gold'`).
Added `_normalize_h5_value()` and updated `recursively_load_dict_contents_from_group()`
to decode byte-string scalars and arrays to Python `str`. Public `h5py.Dataset` /
`h5py.Group` type hints replace the removed private `h5py._hl.*` paths.

### Deprecated NumPy string dtype (`recon_format.py`, `NPY2H5.py`)

`np.string_` (removed in recent NumPy) → `np.bytes_` for HDF5 material name datasets.

---

## Packaging and Dependencies

### `setup.py`

- `distutils` → `setuptools` with `find_packages()`.
- Removed invalid `package_dir={'': ''}` that broke editable installs (`pip install -e .`).

### `pyproject.toml` (new)

PEP 517/518 build metadata for modern `pip install -e .` without invoking
`setup.py` directly.

### `requirements.txt`

Updated minimum versions for Python 3.13 compatibility (NumPy, SciPy, PyCUDA, h5py,
OpenCV, etc.).

### `scripts/recon.py`

Removed a hardcoded developer machine path.

### `makefile`

Minor update to align with the new install flow.

### `README.md`

Added Python 3.13+ / CUDA 12+ install instructions and a note about the texture
migration.

---

## Files Changed

| File | Change |
|------|--------|
| `hexomap/kernel_cuda/device_code.cu` | Texture refs → device memory pointers |
| `hexomap/reconstruction.py` | GPU data upload, texture removal, `_gpu_select_orientations()` |
| `hexomap/cuorientations.py` | Lazy kernel compilation |
| `hexomap/utility.py` | HDF5 byte-string normalization |
| `hexomap/orientation.py` | Dataclass `default_factory` |
| `hexomap/orientation_legacy.py` | Same dataclass fix |
| `hexomap/virtualdiffractor.py` | Dataclass `default_factory` |
| `hexomap/recon_format.py` | `np.bytes_` |
| `NPY2H5.py` | `np.bytes_` |
| `hexomap/cuda_texture.py` | Experimental helper (unused by recon path) |
| `setup.py` | setuptools migration |
| `pyproject.toml` | New build config |
| `requirements.txt` | Updated pins |
| `hexomap_env_py313.yml` | New conda environment |
| `README.md` | Updated install docs |
| `makefile` | Install alignment |
| `scripts/recon.py` | Removed hardcoded path |

---

## Known Issues and Workarounds

### NVCC `compiler-bindir` warning

```
nvcc warning : incompatible redefinition for option 'compiler-bindir'
```

Harmless. Caused by overlapping NVCC flags from conda and `PYCUDA_DEFAULT_NVCC_FLAGS`.

### `surface_functions.h` from other PyCUDA JIT paths

If a future code path triggers PyCUDA elementwise JIT compilation (e.g. other
`gpuarray` operations that compile kernels on the fly), the same header error may
reappear. Workarounds:

1. Upgrade PyCUDA: `pip install --upgrade pycuda`
2. Patch the installed header:
   ```bash
   sed -i 's/#include <surface_functions.h>/\/\/ removed for CUDA 12+/g' \
     $CONDA_PREFIX/lib/python*/site-packages/pycuda/cuda/pycuda-helpers.hpp
   ```

### Performance

Direct device-memory reads replace texture cache for experimental data and reciprocal
vectors. On modern GPUs this is generally fine; if profiling shows a regression,
bindless texture objects could be revisited with a more stable binding layer.

---

## Migration from v0.2

1. Create a new conda environment from `hexomap_env_py313.yml` (do not reuse the old
   Python 3.9 / CUDA 11.5 env).
2. `pip install -e .` from the `v0.3` branch.
3. Existing `.yml`, `.h5`, and binary reduced-data configs are unchanged.
4. Run `python -m hexomap` or `scripts/recon.py --config <your_config>` as before.

No changes to user-written config files or reduced data formats are required.

---

## Error Diagnosis Quick Reference

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `surface_functions.h: No such file or directory` | PyCUDA JIT kernel on CUDA 12+ | Pull latest `v0.3` (includes `_gpu_select_orientations` fix) |
| `ValueError: mutable default ... for field e1` | Python 3.14 dataclass strictness | Pull latest `v0.3` (`default_factory` fix) |
| `sample` reads as `b'gold'` | HDF5 byte strings | Pull latest `v0.3` (`_normalize_h5_value`) |
| `distutils` / `egg_base` install error | Old `setup.py` | Pull latest `v0.3` + use `pip install -e .` |
| `exp data loaded` then CUDA compile error | Not missing data — PyCUDA/CUDA mismatch | See `surface_functions.h` row above |
| Texture / `tex2D` / `tex3D` errors | CUDA 12+ removed texture refs | Pull latest `v0.3` (device memory kernels) |

---

*Generated for HEXOMAP v0.3 modernization (Python 3.13+, CUDA 12/13).*
