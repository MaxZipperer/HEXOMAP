#!python
'''
usage: recon_CHESS.py [-h] -c CONFIG [-g GPU] [-r RECONSTRUCTOR_CONFIG] [--KStart KSTART] [--KEnd KEND] [--KStep KSTEP]

single thread hedm reconstruction, CHESS geometry (single detector, box beam, beam at center of detector)

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file, .yml ,.yaml, h5, hdf5
  -g GPU, --gpu GPU     which gpu to use(0-4)
  --KStart, --KEnd, --KStep
                        beam K range in pixel, if not given, reconstruct only at detK in config

config differences from APS (see scripts/recon.py):
    'etalimit' : np.pi,
    'NDet' : 1,
    'detOmegaStart' : -np.pi,
    'detOmegaRange' : 2 * np.pi,
    'detNJ', 'detNK', 'detPixelJ', 'detPixelK', 'fileBinDetIdx': length 1
    'detL', 'detJ', 'detK': shape [1,1], 'detRot': shape [1,1,3]
'''
import sys
import numpy as np
import os
import hexomap
from hexomap import reconstruction_CHESS  # g-force caller
from hexomap import MicFileTool     # io for reconstruction rst
from hexomap import IntBin          # io for binary image (reduced data)
from hexomap import config
import argparse

def main():
    parser = argparse.ArgumentParser(description='single thread hedm reconstruction, CHESS geometry')
    parser.add_argument('-c','--config', help='config file, .yml ,.yaml, h5, hdf5', default="no config")
    parser.add_argument('-g','--gpu', help='which gpu to use(0-4)', default="0")
    parser.add_argument('-r','--reconstructor_config', help='reconstructor config file, .yml ,.yaml, h5, hdf5', default="no config")
    parser.add_argument('--KStart', help='lower bound of beam K, pixel', type=float, default=None)
    parser.add_argument('--KEnd', help='upper bound of beam K, pixel', type=float, default=None)
    parser.add_argument('--KStep', help='step of beam K, pixel', type=float, default=None)
    args = vars(parser.parse_args())

    gpu=args['gpu']
    print(gpu, args['config'])
    if args['config'].endswith(('.yml','.yaml','h5','hdf5')):
        c = config.Config().load(args['config'])
        print(c)
        print(f"===== loaded external config file: {args['config']}  =====")
    else:
        raise ValueError('CHESS reconstruction needs a config file, see usage')
    if args['reconstructor_config'].endswith(('.yml','.yaml','h5','hdf5')):
        c_reconstructor = config.Config().load(args['reconstructor_config'])
        print(c_reconstructor)
        print(f"===== loaded external reconstructor config file: {args['reconstructor_config']}  =====")
    else:
        c_reconstructor = None
        print('============  loaded default reconstructor config ===================')
    ################# reconstruction #########################
    S = reconstruction_CHESS.Reconstructor_GPU_CHESS(gpuID=gpu)  # each run should contain just one reconstructor instance, other wise GPU memory may not be released correctly.
    if c_reconstructor is not None:
        S.load_reconstructor_config(c_reconstructor)
    S.load_config(c)
    if args['KStart'] is None or args['KEnd'] is None or args['KStep'] is None:
        S.serial_recon_multi_stage()
    else:
        S.serial_recon_multi_K(args['KStart'], args['KEnd'], args['KStep'])
    ################# visualization #########################
    #MicFileTool.plot_mic_and_conf(S.squareMicData, 0.5)

if __name__=="__main__":
    main()
