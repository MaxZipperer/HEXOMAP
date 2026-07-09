#!python
"""
Multi-GPU HEXOMAP reconstruction without MPI.

Spatially tiles the reconstruction region across independent worker processes
(one CUDA context per GPU), runs serial_recon_multi_stage on each tile, merges
results by hit-ratio confidence, then runs a final post-process pass on GPU 0.

Workers are launched as separate Python subprocesses so the parent never
initializes CUDA (which would otherwise serialize GPU access).

Example:
    python scripts/recon_multigpu.py --config examples/ConfigExample.yml
    python scripts/recon_multigpu.py -c my_config.h5 -n 4 --gpus 0,1,2,3
    python scripts/recon_multigpu.py -c my_config.yml -n 3 --mode horizontal --halo 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Sequence

import numpy as np


DEFAULT_DEMO_CONFIG = {
    'micsize': np.array([20, 20]),
    'micVoxelSize': 0.01,
    'micShift': np.array([0.0, 0.0, 0.0]),
    'micMask': None,
    'expdataNDigit': 6,
    'energy': 65.351,
    'sample': 'gold',
    'maxQ': 9,
    'etalimit': 81 / 180.0 * np.pi,
    'NRot': 180,
    'NDet': 2,
    'searchBatchSize': 6000,
    'reverseRot': True,
    'detNJ': np.array([2048, 2048]),
    'detNK': np.array([2048, 2048]),
    'detPixelJ': np.array([0.00148, 0.00148]),
    'detPixelK': np.array([0.00148, 0.00148]),
    'detL': np.array([[4.53571404, 6.53571404]]),
    'detJ': np.array([[1010.79405782, 1027.43844558]]),
    'detK': np.array([[2015.95118521, 2014.30163539]]),
    'detRot': np.array([[[89.48560133, 89.53313565, -0.50680978],
                         [89.42516322, 89.22570012, -0.45511278]]]),
    'fileBin': None,
    'fileBinDigit': 6,
    'fileBinDetIdx': np.array([0, 1]),
    'fileBinLayerIdx': 0,
    '_initialString': 'demo_gold_multigpu',
}


def _normalize_base_mask(mask, imgsize: Sequence[int]) -> np.ndarray:
    imgsize = tuple(int(x) for x in imgsize)
    if mask is None:
        return np.ones(imgsize, dtype=bool)
    if isinstance(mask, str) and mask == 'None':
        return np.ones(imgsize, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != imgsize:
        raise ValueError(f"micMask shape {mask.shape} does not match micsize {imgsize}")
    return mask


def _tile_from_bounds(
    imgsize: Sequence[int],
    base_mask: np.ndarray,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
) -> np.ndarray:
    tile = np.zeros(imgsize, dtype=bool)
    tile[row0:row1, col0:col1] = base_mask[row0:row1, col0:col1]
    return tile


def _apply_halo(
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    imgsize: Sequence[int],
    halo: int,
    extend_rows: tuple[bool, bool],
    extend_cols: tuple[bool, bool],
) -> tuple[int, int, int, int]:
    nx, ny = imgsize
    if extend_rows[0]:
        row0 = max(0, row0 - halo)
    if extend_rows[1]:
        row1 = min(nx, row1 + halo)
    if extend_cols[0]:
        col0 = max(0, col0 - halo)
    if extend_cols[1]:
        col1 = min(ny, col1 + halo)
    return row0, row1, col0, col1


def gen_tile_masks(
    imgsize: Sequence[int],
    n_workers: int,
    mask: Optional[np.ndarray] = None,
    halo: int = 5,
    mode: str = 'auto',
) -> List[np.ndarray]:
    """Build per-worker boolean masks that tile the reconstruction region."""
    imgsize = tuple(int(x) for x in imgsize)
    base_mask = _normalize_base_mask(mask, imgsize)

    if n_workers <= 1:
        return [base_mask.copy()]

    if mode == 'auto':
        root = int(round(np.sqrt(n_workers)))
        mode = 'grid' if root * root == n_workers else 'horizontal'

    if mode == 'horizontal':
        nx, ny = imgsize
        bounds = [0]
        for i in range(1, n_workers):
            bounds.append(i * nx // n_workers)
        bounds.append(nx)

        tiles = []
        for i in range(n_workers):
            row0, row1 = bounds[i], bounds[i + 1]
            row0, row1, col0, col1 = _apply_halo(
                row0, row1, 0, ny, imgsize, halo,
                extend_rows=(i > 0, i < n_workers - 1),
                extend_cols=(False, False),
            )
            tiles.append(_tile_from_bounds(imgsize, base_mask, row0, row1, col0, col1))
        return tiles

    if mode == 'grid':
        nx_grid = int(np.ceil(np.sqrt(n_workers)))
        ny_grid = int(np.ceil(n_workers / nx_grid))
        nx, ny = imgsize

        row_bounds = [i * nx // nx_grid for i in range(nx_grid)] + [nx]
        col_bounds = [j * ny // ny_grid for j in range(ny_grid)] + [ny]

        tiles = []
        worker_idx = 0
        for i in range(nx_grid):
            for j in range(ny_grid):
                if worker_idx >= n_workers:
                    break
                row0, row1 = row_bounds[i], row_bounds[i + 1]
                col0, col1 = col_bounds[j], col_bounds[j + 1]
                row0, row1, col0, col1 = _apply_halo(
                    row0, row1, col0, col1, imgsize, halo,
                    extend_rows=(i > 0, i < nx_grid - 1),
                    extend_cols=(j > 0, j < ny_grid - 1),
                )
                tiles.append(_tile_from_bounds(imgsize, base_mask, row0, row1, col0, col1))
                worker_idx += 1
        if len(tiles) != n_workers:
            raise RuntimeError(
                f"grid tiling produced {len(tiles)} masks for {n_workers} workers"
            )
        return tiles

    raise ValueError(f"Unknown tiling mode: {mode}")


def merge_tile_results(tile_results: Sequence[np.ndarray]) -> np.ndarray:
    """Merge per-tile squareMicData arrays, keeping the higher hit-ratio voxel."""
    if not tile_results:
        raise ValueError("No tile results to merge")
    if len(tile_results) == 1:
        return tile_results[0].copy()

    mic = np.zeros_like(tile_results[0])
    coord = tile_results[0][:, :, 0:3]
    for tile in tile_results:
        better = tile[:, :, 6] > mic[:, :, 6]
        mic[np.repeat(better[:, :, np.newaxis], mic.shape[2], axis=2)] = tile[
            np.repeat(better[:, :, np.newaxis], mic.shape[2], axis=2)
        ]
    mic[:, :, 0:3] = coord
    return mic


def _parse_gpu_ids(gpu_arg: Optional[str], n_workers: int) -> List[int]:
    """
    Resolve GPU ids without initializing CUDA in the parent process.

    Parent CUDA initialization is the most common reason multi-GPU workers run
    serially when using multiprocessing pools with PyCUDA.
    """
    if gpu_arg:
        gpu_ids = [int(x.strip()) for x in gpu_arg.split(',') if x.strip() != '']
        if len(gpu_ids) != n_workers:
            raise ValueError(
                f"--gpus lists {len(gpu_ids)} device(s) but -n requests {n_workers}"
            )
        return gpu_ids
    return list(range(n_workers))


def _load_config(
    config_path: Optional[str],
    reconstructor_config_path: Optional[str],
):
    import hexomap
    from hexomap import config

    if config_path and config_path.endswith(('.yml', '.yaml', '.h5', '.hdf5')):
        c = config.Config().load(config_path)
    else:
        demo = dict(DEFAULT_DEMO_CONFIG)
        demo['fileBin'] = os.path.abspath(
            os.path.join(os.path.dirname(hexomap.__file__), "..",
                         "examples/johnson_aug18_demo/Au_reduced_1degree/Au_int_1degree_suter_aug18_z")
        )
        c = config.Config(**demo)

    c_reconstructor = None
    if reconstructor_config_path and reconstructor_config_path.endswith(
        ('.yml', '.yaml', '.h5', '.hdf5')
    ):
        c_reconstructor = config.Config().load(reconstructor_config_path)
    return c, c_reconstructor


def _recon_worker(
    worker_id: int,
    gpu_id: int,
    config_path: Optional[str],
    reconstructor_config_path: Optional[str],
    tile_mask_path: str,
    initial_string: str,
    output_path: str,
    enable_post_process: bool,
) -> None:
    """Run serial reconstruction for one spatial tile on a dedicated GPU."""
    import pycuda.driver as cuda
    from hexomap import config, reconstruction

    tile_mask = np.load(tile_mask_path)

    cuda.init()
    ctx = cuda.Device(gpu_id).make_context()
    try:
        if config_path and config_path.endswith(('.yml', '.yaml', '.h5', '.hdf5')):
            c = config.Config().load(config_path)
        else:
            import hexomap
            demo = dict(DEFAULT_DEMO_CONFIG)
            demo['fileBin'] = os.path.abspath(
                os.path.join(os.path.dirname(hexomap.__file__), "..",
                             "examples/johnson_aug18_demo/Au_reduced_1degree/Au_int_1degree_suter_aug18_z")
            )
            c = config.Config(**demo)

        c_reconstructor = None
        if reconstructor_config_path and reconstructor_config_path.endswith(
            ('.yml', '.yaml', '.h5', '.hdf5')
        ):
            c_reconstructor = config.Config().load(reconstructor_config_path)

        c.micMask = tile_mask
        c._initialString = f"{initial_string}_part_{worker_id}"

        started = time.time()
        print(f"[worker {worker_id}] GPU {gpu_id}: pid={os.getpid()}, "
              f"{int(np.sum(tile_mask))} voxels in tile", flush=True)

        recon = reconstruction.Reconstructor_GPU(ctx=ctx)
        if c_reconstructor is not None:
            recon.load_reconstructor_config(c_reconstructor)
        recon.load_config(c)
        recon.serial_recon_multi_stage(enablePostProcess=enable_post_process)

        np.save(output_path, recon.squareMicData)
        elapsed = time.time() - started
        print(f"[worker {worker_id}] GPU {gpu_id}: finished in {elapsed:.1f}s, "
              f"saved {output_path}", flush=True)
    finally:
        ctx.pop()


def _final_postprocess(
    gpu_id: int,
    config_path: Optional[str],
    reconstructor_config_path: Optional[str],
    merged_mic: np.ndarray,
    base_mask: Optional[np.ndarray],
    initial_string: str,
) -> None:
    """Merge-aware cleanup pass on a single GPU."""
    import pycuda.driver as cuda
    from hexomap import reconstruction

    cuda.init()
    ctx = cuda.Device(gpu_id).make_context()
    try:
        c, c_reconstructor = _load_config(config_path, reconstructor_config_path)
        c._initialString = initial_string
        c.micMask = base_mask

        recon = reconstruction.Reconstructor_GPU(ctx=ctx)
        if c_reconstructor is not None:
            recon.load_reconstructor_config(c_reconstructor)
        # Must upload experimental data: post_process calls GPU hitratio kernels.
        recon.load_config(c, reloadData=True)
        recon.load_square_mic(merged_mic)
        recon.voxelIdxStage0 = []
        recon.serial_recon_multi_stage(enablePostProcess=True)
    finally:
        ctx.pop()


def _launch_worker_subprocess(
    script_path: str,
    worker_id: int,
    gpu_id: int,
    config_path: Optional[str],
    reconstructor_config_path: Optional[str],
    tile_mask_path: str,
    initial_string: str,
    output_path: str,
    enable_post_process: bool,
) -> subprocess.Popen:
    cmd = [
        sys.executable, '-u', script_path,
        '--_worker',
        '--worker-id', str(worker_id),
        '--gpu-id', str(gpu_id),
        '--tile-mask', tile_mask_path,
        '--output', output_path,
        '--initial-string', initial_string,
    ]
    if not enable_post_process:
        cmd.append('--no-tile-postprocess')
    if config_path:
        cmd.extend(['-c', config_path])
    if reconstructor_config_path:
        cmd.extend(['-r', reconstructor_config_path])

    return subprocess.Popen(cmd)


def _worker_main(args: argparse.Namespace) -> int:
    _recon_worker(
        worker_id=args.worker_id,
        gpu_id=args.gpu_id,
        config_path=args.config if args.config != 'no config' else None,
        reconstructor_config_path=(
            args.reconstructor_config
            if args.reconstructor_config != 'no config'
            else None
        ),
        tile_mask_path=args.tile_mask,
        initial_string=args.initial_string,
        output_path=args.output,
        enable_post_process=not args.no_tile_postprocess,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-GPU HEXOMAP reconstruction. "
            "Tiles the microstructure across GPUs and merges by hit ratio."
        )
    )
    parser.add_argument(
        '-c', '--config',
        help='Config file (.yml, .yaml, .h5, .hdf5)',
        default='no config',
    )
    parser.add_argument(
        '-r', '--reconstructor_config',
        help='Optional reconstructor config file',
        default='no config',
    )
    parser.add_argument(
        '-n', '--ngpus',
        type=int,
        default=2,
        help='Number of GPU workers (default: 2)',
    )
    parser.add_argument(
        '--gpus',
        help='Comma-separated GPU ids, one per worker (default: 0..n-1)',
        default=None,
    )
    parser.add_argument(
        '--mode',
        choices=['auto', 'horizontal', 'grid'],
        default='auto',
        help='Spatial tiling strategy (default: auto)',
    )
    parser.add_argument(
        '--halo',
        type=int,
        default=5,
        help='Voxel overlap between adjacent tiles (default: 5)',
    )
    parser.add_argument(
        '--keep-tiles',
        action='store_true',
        help='Keep per-tile .npy outputs in a temp directory',
    )
    parser.add_argument(
        '--no-final-pass',
        action='store_true',
        help='Skip the merged post-process pass on GPU 0',
    )
    parser.add_argument(
        '--no-tile-postprocess',
        action='store_true',
        help='Disable post-process inside each tile worker',
    )
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-id', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--gpu-id', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--tile-mask', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    parser.add_argument('--initial-string', default='hexomap', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args._worker:
        return _worker_main(args)

    if args.ngpus < 1:
        parser.error("--ngpus must be at least 1")

    config_path = args.config if args.config != 'no config' else None
    reconstructor_config_path = (
        args.reconstructor_config
        if args.reconstructor_config != 'no config'
        else None
    )

    c, _ = _load_config(config_path, reconstructor_config_path)
    print(c)

    base_mask = None
    try:
        base_mask = c.micMask
        if isinstance(base_mask, str) and base_mask == 'None':
            base_mask = None
    except AttributeError:
        base_mask = None

    initial_string = c._initialString
    gpu_ids = _parse_gpu_ids(args.gpus, args.ngpus)
    tile_masks = gen_tile_masks(
        c.micsize,
        args.ngpus,
        mask=base_mask,
        halo=args.halo,
        mode=args.mode,
    )

    print(f"Tiling {c.micsize} across {args.ngpus} worker(s) on GPU(s) {gpu_ids} "
          f"(mode={args.mode}, halo={args.halo})")
    for i, tile in enumerate(tile_masks):
        print(f"  tile {i}: {int(np.sum(tile))} voxels")

    script_path = os.path.abspath(__file__)
    tmp_dir = tempfile.mkdtemp(prefix='hexomap_multigpu_')
    mask_paths = [
        os.path.join(tmp_dir, f'tile_mask_{i}.npy') for i in range(args.ngpus)
    ]
    output_paths = [
        os.path.join(tmp_dir, f'tile_{i}.npy') for i in range(args.ngpus)
    ]
    for mask, path in zip(tile_masks, mask_paths):
        np.save(path, mask)

    wall_start = time.time()
    if args.ngpus == 1:
        _recon_worker(
            0,
            gpu_ids[0],
            config_path,
            reconstructor_config_path,
            mask_paths[0],
            initial_string,
            output_paths[0],
            not args.no_tile_postprocess,
        )
        tile_elapsed = [time.time() - wall_start]
    else:
        print(f"Launching {args.ngpus} parallel worker subprocesses ...", flush=True)
        processes = [
            _launch_worker_subprocess(
                script_path,
                i,
                gpu_ids[i],
                config_path,
                reconstructor_config_path,
                mask_paths[i],
                initial_string,
                output_paths[i],
                not args.no_tile_postprocess,
            )
            for i in range(args.ngpus)
        ]
        exit_codes = [proc.wait() for proc in processes]
        tile_elapsed = [time.time() - wall_start]
        if any(code != 0 for code in exit_codes):
            raise RuntimeError(
                f"One or more workers failed with exit codes: {exit_codes}"
            )

    wall_tile_time = time.time() - wall_start
    print(f"All tile workers finished in {wall_tile_time:.1f}s wall time", flush=True)

    tile_results = [np.load(path) for path in output_paths]
    merged_mic = merge_tile_results(tile_results)
    merged_path = os.path.join(tmp_dir, 'merged_square_mic.npy')
    np.save(merged_path, merged_mic)
    print(f"Merged tile results saved to {merged_path}")

    if not args.no_final_pass:
        print(f"Running final post-process on GPU {gpu_ids[0]} ...", flush=True)
        final_start = time.time()
        _final_postprocess(
            gpu_ids[0],
            config_path,
            reconstructor_config_path,
            merged_mic,
            base_mask,
            initial_string,
        )
        print(f"Final post-process complete in {time.time() - final_start:.1f}s.")

    if args.keep_tiles:
        print(f"Per-tile outputs kept in {tmp_dir}")
    else:
        for path in output_paths + mask_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        print(f"Temporary tile files removed; merged output kept at {merged_path}")

    print(f"Total wall time: {time.time() - wall_start:.1f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
