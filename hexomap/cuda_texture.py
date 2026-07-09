"""
CUDA texture object helpers for PyCUDA.

PyCUDA only wraps the legacy texture-reference API, which was removed in
CUDA 12+.  These helpers create bindless texture objects via the CUDA driver
API and return 64-bit handles that can be passed directly to kernels.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Callable, Optional, TypeVar

import numpy as np
import pycuda.driver as cuda

T = TypeVar("T")

CUresult = ctypes.c_int
CUtexObject = ctypes.c_uint64

CU_RESOURCE_TYPE_ARRAY = 0x00

CU_TR_ADDRESS_MODE_WRAP = 0
CU_TR_FILTER_MODE_POINT = 0
CU_TRSF_READ_AS_INTEGER = 0x01


class _RES_ARRAY(ctypes.Structure):
    _fields_ = [("hArray", ctypes.c_void_p)]


class _RES_MIPMAP(ctypes.Structure):
    _fields_ = [("hMipmappedArray", ctypes.c_void_p)]


class _RES_LINEAR(ctypes.Structure):
    _fields_ = [
        ("devPtr", ctypes.c_uint64),
        ("format", ctypes.c_int),
        ("numChannels", ctypes.c_uint32),
        ("sizeInBytes", ctypes.c_size_t),
    ]


class _RES_PITCH2D(ctypes.Structure):
    _fields_ = [
        ("devPtr", ctypes.c_uint64),
        ("format", ctypes.c_int),
        ("numChannels", ctypes.c_uint32),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("pitchInBytes", ctypes.c_size_t),
    ]


class _RES_UNION(ctypes.Union):
    _fields_ = [
        ("array", _RES_ARRAY),
        ("mipmap", _RES_MIPMAP),
        ("linear", _RES_LINEAR),
        ("pitch2D", _RES_PITCH2D),
    ]


class CUDA_RESOURCE_DESC(ctypes.Structure):
    _fields_ = [
        ("resType", ctypes.c_uint32),
        ("res", _RES_UNION),
        ("flags", ctypes.c_uint32),
    ]


class CUDA_TEXTURE_DESC(ctypes.Structure):
    _fields_ = [
        ("addressMode", ctypes.c_int * 3),
        ("filterMode", ctypes.c_int),
        ("flags", ctypes.c_uint32),
        ("maxAnisotropy", ctypes.c_uint32),
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
    ctypes.c_void_p,
]
_cuda.cuTexObjectCreate.restype = CUresult
_cuda.cuTexObjectDestroy.argtypes = [CUtexObject]
_cuda.cuTexObjectDestroy.restype = CUresult
_cuda.cuGetErrorString.argtypes = [CUresult]
_cuda.cuGetErrorString.restype = ctypes.c_char_p


def _check_cu(result: int, msg: str) -> None:
    if result != 0:
        err = _cuda.cuGetErrorString(result)
        detail = err.decode() if err else f"CUDA driver error {result}"
        raise RuntimeError(f"{msg} ({detail})")


def _array_handle(array: cuda.Array) -> int:
    handle = array.handle
    return int(handle) if not isinstance(handle, int) else handle


def _create_texture_from_array(
    array: cuda.Array,
    *,
    read_as_integer: bool = False,
) -> int:
    res_desc = CUDA_RESOURCE_DESC()
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY
    res_desc.res.array.hArray = ctypes.c_void_p(_array_handle(array))
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

    tex_obj = CUtexObject(0)
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

    def __init__(self, ctx: Optional[cuda.Context] = None) -> None:
        self._ctx = ctx
        self._array: Optional[cuda.Array] = None
        self._handle: Optional[int] = None

    def _get_ctx(self) -> cuda.Context:
        ctx = self._ctx or cuda.Context.get_current()
        if ctx is None:
            raise RuntimeError("No active CUDA context for texture operations")
        return ctx

    def _run_in_context(self, fn: Callable[[], T]) -> T:
        ctx = self._get_ctx()
        ctx.push()
        try:
            return fn()
        finally:
            ctx.pop()

    def bind_2d_float(self, host_array: np.ndarray, order: str = "C") -> None:
        def _bind() -> None:
            self.release()
            self._array = cuda.np_to_array(host_array.astype(np.float32), order=order)
            self._handle = _create_texture_from_array(self._array, read_as_integer=False)

        self._run_in_context(_bind)

    def bind_3d_uint8(self, host_array: np.ndarray, order: str = "C") -> None:
        def _bind() -> None:
            self.release()
            self._array = cuda.np_to_array(host_array.astype(np.uint8), order=order)
            self._handle = _create_texture_from_array(self._array, read_as_integer=True)

        self._run_in_context(_bind)

    def as_kernel_arg(self) -> np.uint64:
        if self._handle is None:
            raise RuntimeError("texture object is not bound")
        return np.uint64(self._handle)

    def release(self) -> None:
        """Destroy the texture object. Safe to call multiple times."""
        if self._handle is None:
            self._array = None
            return
        handle = self._handle
        self._handle = None
        array = self._array
        self._array = None
        try:
            ctx = self._get_ctx()
        except RuntimeError:
            return
        ctx.push()
        try:
            err = _cuda.cuTexObjectDestroy(CUtexObject(handle))
            if err != 0:
                _check_cu(err, "cuTexObjectDestroy failed")
        except Exception:
            pass
        finally:
            ctx.pop()
        del array
