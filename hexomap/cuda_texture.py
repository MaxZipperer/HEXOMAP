"""
CUDA texture object helpers for PyCUDA.

PyCUDA only wraps the legacy texture-reference API, which was removed in
CUDA 12+.  These helpers create bindless texture objects via the CUDA driver
API and return 64-bit handles that can be passed directly to kernels.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Optional

import numpy as np
import pycuda.driver as cuda

CUresult = ctypes.c_int
CUtexObject = ctypes.c_uint64
CUarray = ctypes.c_void_p

CU_RESOURCE_TYPE_ARRAY = 0x00

CU_TR_ADDRESS_MODE_WRAP = 0
CU_TR_FILTER_MODE_POINT = 0
CU_TR_READ_MODE_ELEMENT_TYPE = 0
CU_TR_READ_MODE_INTEGER = 1

CU_TRSF_READ_AS_INTEGER = 0x01


class _CUDA_RESOURCE_DESC_ARRAY(ctypes.Structure):
    _fields_ = [("hArray", CUarray)]


class _CUDA_RESOURCE_DESC_UNION(ctypes.Structure):
    _fields_ = [
        ("array", _CUDA_RESOURCE_DESC_ARRAY),
        ("devPtr", ctypes.c_uint64),
        ("pitch2D", ctypes.c_byte * 32),
        ("mipmap", ctypes.c_byte * 16),
    ]


class CUDA_RESOURCE_DESC(ctypes.Structure):
    _fields_ = [
        ("resType", ctypes.c_int),
        ("res", _CUDA_RESOURCE_DESC_UNION),
        ("flags", ctypes.c_uint),
    ]


class CUDA_TEXTURE_DESC(ctypes.Structure):
    _fields_ = [
        ("addressMode", ctypes.c_int * 3),
        ("filterMode", ctypes.c_int),
        ("flags", ctypes.c_uint),
        ("maxAnisotropy", ctypes.c_uint),
        ("mipmapFilterMode", ctypes.c_int),
        ("mipmapLevelBias", ctypes.c_float),
        ("minMipmapLevelClamp", ctypes.c_float),
        ("maxMipmapLevelClamp", ctypes.c_float),
        ("borderColor", ctypes.c_float * 4),
        ("reserved", ctypes.c_int * 12),
    ]


def _load_cuda_driver():
    if sys.platform == "win32":
        return ctypes.windll.nvcuda
    if sys.platform == "darwin":
        return ctypes.CDLL("libcuda.dylib")
    return ctypes.CDLL("libcuda.so.1")


_cuda = _load_cuda_driver()
_cuda.cuTexObjectCreate.argtypes = [
    ctypes.POINTER(CUtexObject),
    ctypes.POINTER(CUDA_RESOURCE_DESC),
    ctypes.POINTER(CUDA_TEXTURE_DESC),
    ctypes.POINTER(CUDA_RESOURCE_VIEW_DESC),
]
_cuda.cuTexObjectCreate.restype = CUresult
_cuda.cuTexObjectDestroy.argtypes = [CUtexObject]
_cuda.cuTexObjectDestroy.restype = CUresult


def _check_cu(result: int, msg: str) -> None:
    if result != 0:
        raise RuntimeError(f"{msg} (CUDA driver error {result})")


def _create_texture_from_array(
    array: cuda.Array,
    *,
    read_as_integer: bool = False,
) -> int:
    res_desc = CUDA_RESOURCE_DESC()
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY
    res_desc.res.array.hArray = CUarray(int(array.handle))
    res_desc.flags = 0

    tex_desc = CUDA_TEXTURE_DESC()
    tex_desc.addressMode[:] = (CU_TR_ADDRESS_MODE_WRAP,) * 3
    tex_desc.filterMode = CU_TR_FILTER_MODE_POINT
    tex_desc.flags = CU_TRSF_READ_AS_INTEGER if read_as_integer else 0
    tex_desc.maxAnisotropy = 1
    tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT
    tex_desc.mipmapLevelBias = 0.0
    tex_desc.minMipmapLevelClamp = 0.0
    tex_desc.maxMipmapLevelClamp = 0.0

    tex_obj = CUtexObject()
    err = _cuda.cuTexObjectCreate(
        ctypes.byref(tex_obj),
        ctypes.byref(res_desc),
        ctypes.byref(tex_desc),
        None,
    )
    _check_cu(err, "cuTexObjectCreate failed")
    return int(tex_obj.value)


class CudaTextureObject:
    """Manage a CUDA texture object backed by a PyCUDA Array."""

    def __init__(self) -> None:
        self._array: Optional[cuda.Array] = None
        self._handle: Optional[int] = None

    def bind_2d_float(self, host_array: np.ndarray, order: str = "C") -> None:
        self._release()
        self._array = cuda.np_to_array(host_array.astype(np.float32), order=order)
        self._handle = _create_texture_from_array(self._array, read_as_integer=False)

    def bind_3d_uint8(self, host_array: np.ndarray, order: str = "C") -> None:
        self._release()
        self._array = cuda.np_to_array(host_array.astype(np.uint8), order=order)
        self._handle = _create_texture_from_array(self._array, read_as_integer=True)

    def as_kernel_arg(self) -> np.uint64:
        if self._handle is None:
            raise RuntimeError("texture object is not bound")
        return np.uint64(self._handle)

    def _release(self) -> None:
        if self._handle is not None:
            _cuda.cuTexObjectDestroy(CUtexObject(self._handle))
            self._handle = None
        self._array = None

    def __del__(self) -> None:
        try:
            self._release()
        except Exception:
            pass
