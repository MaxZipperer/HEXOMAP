#!/usr/bin/env python
# CHESS geometry: single detector, box beam, beam at the center of the detector, 360 degree rotation.
# Same reconstruction procedure as reconstruction.Reconstructor_GPU (APS geometry), only the
# diffraction simulation kernel is replaced by simulation_CHESS from kernel_cuda/device_code_CHESS.cu.
import numpy as np
from pycuda.compiler import SourceModule
from hexomap.utility import load_kernel_code
from hexomap.reconstruction import Reconstructor_GPU
import hexomap
import os.path


class Reconstructor_GPU_CHESS(Reconstructor_GPU):
    '''
    example usage:
        S = reconstruction_CHESS.Reconstructor_GPU_CHESS(gpuID=0)
        S.load_config(c)
        S.serial_recon_multi_K(KStart=900, KEnd=1100, KStep=10)
    config differences from APS:
        NDet: 1
        detL, detJ, detK: shape [1,1], detRot: shape [1,1,3]
        etalimit: np.pi, in radian
        detOmegaStart: -np.pi, in radian
        detOmegaRange: 2 * np.pi, in radian
    '''
    def _init_gpu(self):
        super()._init_gpu()
        _kernel_code_CHESS = os.path.join(os.path.dirname(hexomap.__file__),
                                    "kernel_cuda/device_code_CHESS.cu",
                                    )
        self.mod_CHESS = SourceModule(load_kernel_code(_kernel_code_CHESS))
        self.sim_func = self.mod_CHESS.get_function("simulation_CHESS")

    def load_config(self, config, reloadData=True):
        '''
        same as Reconstructor_GPU.load_config, with CHESS defaults:
        etalimit: np.pi, in radian
        detOmegaStart: -np.pi, in radian
        detOmegaRange: 2 * np.pi, in radian
        '''
        if not hasattr(config, 'etalimit'):
            config.etalimit = np.pi
        if not hasattr(config, 'detOmegaStart'):
            config.detOmegaStart = - np.pi
        if not hasattr(config, 'detOmegaRange'):
            config.detOmegaRange = 2 * np.pi
        super().load_config(config, reloadData=reloadData)

    def serial_recon_multi_K(self, KStart, KEnd, KStep, enablePostProcess=True, verbose=True):
        '''
        box beam: treat the beam as a line at each K between KStart and KEnd (pixel, inclusive),
        and do a full serial_recon_multi_stage at each K. K is not refined.
        :param KStart: lower bound of beam K position, pixel
        :param KEnd: upper bound of beam K position, pixel
        :param KStep: step of K, pixel
        :return: list of output file names, one per K
        '''
        c = self.config
        micInitial = self.squareMicData.copy()
        squareMicOutFileInitial = self.squareMicOutFile
        lOutFile = []
        for K in np.arange(KStart, KEnd + 0.5 * KStep, KStep):
            print(f'============ reconstruct at K: {K} ============')
            self.set_det_param(np.array(c.detL), np.array(c.detJ), np.array([[K]]), np.array(c.detRot),
                               NJ=np.array(c.detNJ), NK=np.array(c.detNK),
                               pixelJ=np.array(c.detPixelJ), pixelK=np.array(c.detPixelK))
            self.load_square_mic(micInitial.copy())
            self.squareMicOutFile = squareMicOutFileInitial.replace('.npy', f'_K{K:.2f}.npy')
            self.serial_recon_multi_stage(enablePostProcess=enablePostProcess, verbose=verbose)
            lOutFile.append(self.squareMicOutFile)
        self.squareMicOutFile = squareMicOutFileInitial
        return lOutFile

    # geometry optimization assumes APS geometry (2 detectors, line beam, refining K), not usable for CHESS
    def blind_search_parameter(self, *args, **kwargs):
        raise NotImplementedError('blind_search_parameter only supports APS geometry')

    def geo_opt_coordinate_search(self, *args, **kwargs):
        raise NotImplementedError('geo_opt_coordinate_search only supports APS geometry')

    def twiddle_refine(self, *args, **kwargs):
        raise NotImplementedError('twiddle_refine only supports APS geometry')

    def auto_twiddle(self, *args, **kwargs):
        raise NotImplementedError('auto_twiddle only supports APS geometry')
