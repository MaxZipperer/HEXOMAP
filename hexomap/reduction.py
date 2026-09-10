import cv2
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndi
import sys
import tifffile as tiff
import time
import yaml

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from numba import jit
from numba import njit
from scipy.ndimage import label

import hexomap
from hexomap import IntBin

def iter_image_block(layer, det, rstart, numFiles, config):
    '''
    Yield rotations rstart to rstart+numFiles-1 of one detector one image at a
    time, so that a caller that works frame by frame never holds the whole run.

    With imgFormat 'h5' the images come from one rotation stack per recorded
    file, named imgPrefix + (startIdx + layer*NDet + det) + extension, or from a
    single archive holding every detector when h5Img is set.  Otherwise each
    rotation is its own file, numbered startIdx + (layer*NDet + det)*NRot + rot.
    '''
    iDet = list(config['layerIdx']).index(layer)*config['NDet'] + det

    if str(config.get('imgFormat', 'tif')).lower() in ('h5', 'hdf5'):
        fName = config.get('h5Img') or (f"{config['imgPrefix']}"
                                        f"{str(config['startIdx'] + iDet).zfill(config['NDigit'])}"
                                        f"{config['extension']}")
        #a single archive holds the detectors one NRot long stretch after another
        istart = (int(config.get('h5StartFrame', 0)) + rstart
                  + (iDet*config['NRot'] if config.get('h5Img') else 0))
        with h5py.File(fName, 'r') as fin:
            lPath = [config['h5Path']] if config.get('h5Path') else []
            if not lPath:
                #fall back to the only 3D dataset in the archive
                fin.visititems(lambda name, obj: lPath.append(name)
                               if isinstance(obj, h5py.Dataset) and obj.ndim == 3 else None)
                if len(lPath) != 1:
                    raise KeyError(f"Cannot choose an image stack among {lPath}, set config['h5Path']")
            dset = fin[lPath[0]]
            if tuple(dset.shape[1:]) != (config['dety'], config['detz']):
                raise ValueError(f"{fName}: images are {dset.shape[1:]}, config says {(config['dety'], config['detz'])}")
            if istart + numFiles > dset.shape[0]:
                raise IndexError(f"{fName}: need images {istart}-{istart + numFiles - 1} but the stack only holds {dset.shape[0]}")
            for i in range(numFiles):
                yield dset[istart + i]
    else:
        for i in range(numFiles):
            ifile = config['startIdx'] + iDet*config['NRot'] + rstart + i
            try:
                yield tiff.imread(f"{config['imgPrefix']}{str(ifile).zfill(config['NDigit'])}{config['extension']}")
            except Exception as e:
                print(f"Error reading file {ifile}: {e}")
                raise

def compute_median_block(ilayer, idet, rstart, numFiles, medFile, config):
    '''
    Median of one block of rotations, saved for the reduction pass to pick up.
    Holds the whole block, which is unavoidable for a median over rotations, and
    releases it as soon as the median is written.
    '''
    t0 = time.perf_counter()
    raw = np.empty((numFiles, config['dety'], config['detz']), dtype=np.int16)
    for i, img in enumerate(iter_image_block(ilayer, idet, rstart, numFiles, config)):
        raw[i, :, :] = img
    t1 = time.perf_counter()

    #overwrite_input lets numpy partition the block in place rather than copy it,
    #which is safe here because the reduction pass rereads the images anyway
    np.save(medFile, np.median(raw, axis=0, overwrite_input=True).astype(np.int16))
    return (f"[L{ilayer} D{idet}] median over rotations {rstart}-{rstart + numFiles - 1}"
            f" | Read: {t1-t0:.2f}s | Median: {time.perf_counter()-t1:.2f}s")

@njit(cache=True)
def extract_peaks_numbaV2(img_sub, baseline=0, minNPixel=4):
    #Peak extraction by Zipeng Xu
    h, w = img_sub.shape
    #the flood fill only ever asked whether a pixel was already taken, never
    #which region took it, so the mask doubles as the visited marker
    mask = img_sub > baseline

    # Temporary arrays for collecting region-wise data, sized by the number of
    # pixels above baseline since no other pixel can ever be stored
    nMask = np.count_nonzero(mask)
    lXTmp = np.empty(nMask, dtype=np.int32)
    lYTmp = np.empty(nMask, dtype=np.int32)
    lValTmp = np.empty(nMask, dtype=np.float32)
    lIDTmp = np.empty(nMask, dtype=np.int32)
    lStartIdx = np.empty(nMask + 1, dtype=np.int32)
    lEndIdx = np.empty(nMask + 1, dtype=np.int32)
    lMaxVal = np.empty(nMask + 1, dtype=np.float32)  # store region max intensity
    label_counter = 0
    idx = 0

    # explicit stack, grown on demand, in place of a list of coordinate tuples
    stackX = np.empty(1024, dtype=np.int64)
    stackY = np.empty(1024, dtype=np.int64)

    for x in range(h):
        for y in range(w):
            if mask[x, y]:
                # Start flood-fill
                stackX[0] = x
                stackY[0] = y
                nStack = 1
                region_idx_start = idx

                while nStack > 0:
                    nStack -= 1
                    cx = stackX[nStack]
                    cy = stackY[nStack]
                    if 0 <= cx < h and 0 <= cy < w and mask[cx, cy]:
                        mask[cx, cy] = False
                        lXTmp[idx] = cx
                        lYTmp[idx] = cy
                        lValTmp[idx] = img_sub[cx, cy]
                        lIDTmp[idx] = label_counter
                        idx += 1
                        if nStack + 8 > stackX.size:
                            biggerX = np.empty(stackX.size*2, dtype=np.int64)
                            biggerY = np.empty(stackY.size*2, dtype=np.int64)
                            biggerX[:nStack] = stackX[:nStack]
                            biggerY[:nStack] = stackY[:nStack]
                            stackX = biggerX
                            stackY = biggerY
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                if dx != 0 or dy != 0:
                                    stackX[nStack] = cx + dx
                                    stackY[nStack] = cy + dy
                                    nStack += 1

                region_idx_end = idx

                # Compute maximum value in the region
                vmax = np.float32(0.0)
                for i in range(region_idx_start, region_idx_end):
                    if lValTmp[i] > vmax:
                        vmax = lValTmp[i]

                if vmax > baseline:
                    count_valid = 0
                    for i in range(region_idx_start, region_idx_end):
                        if lValTmp[i] > max(np.float32(1.0), np.float32(0.1)*vmax):
                            count_valid += 1
                    if count_valid >= minNPixel:
                        lStartIdx[label_counter] = region_idx_start
                        lEndIdx[label_counter] = region_idx_end
                        lMaxVal[label_counter] = vmax
                        label_counter += 1
                    else:
                        idx = region_idx_start  # discard region
                else:
                    idx = region_idx_start  # discard region

    # Output final arrays
    total_points = 0
    for i in range(label_counter):
        for j in range(lStartIdx[i], lEndIdx[i]):
            if lValTmp[j] > max(np.float32(1.0), np.float32(0.1)*lMaxVal[i]):
                total_points += 1

    lX = np.empty(total_points, dtype=np.int32)
    lY = np.empty(total_points, dtype=np.int32)
    lVal = np.empty(total_points, dtype=np.int32)
    lID = np.empty(total_points, dtype=np.int32)

    idx_out = 0
    for i in range(label_counter):
        for j in range(lStartIdx[i], lEndIdx[i]):
            if lValTmp[j] > max(np.float32(1.0), np.float32(0.1)*lMaxVal[i]):
                lX[idx_out] = lXTmp[j]
                lY[idx_out] = lYTmp[j]
                lVal[idx_out] = int(lValTmp[j])
                lID[idx_out] = lIDTmp[j]
                idx_out += 1

    X_flipped = (w - 1 - lY[:idx_out]).astype(np.int32)
    Y = lX[:idx_out].astype(np.int32)
    Intensity = lVal[:idx_out].astype(np.int32)
    PeakID = lID[:idx_out].astype(np.int32)

    return X_flipped, Y, Intensity, PeakID

def reduce_frame_chunk(ilayer, idet, rstart, numFiles, medFile, config):
    '''
    Subtract the block median from a run of frames, filter, extract peaks and
    write the binaries.  Only one image is held at a time, so these tasks can be
    spread as widely as there are cores.
    '''
    t0 = time.perf_counter()
    # Default is to print everything out
    verbose = config.get('verbose') is None or config['verbose'] is True
    floor = np.load(medFile) + config['blanket']

    for ind, frame in enumerate(iter_image_block(ilayer, idet, rstart, numFiles, config)):
        frame_index = rstart + ind
        if verbose:
            print(f' \n Reducing image {frame_index}')
        img = frame.astype(np.int16) - floor
        img[img < 0] = 0
        img = img.astype(np.float32)
        LoG = ndi.gaussian_laplace(img, sigma=config['LoGsig'])
        snp = extract_peaks_numbaV2((LoG < config['LoGcut'])*img,
                                    baseline=config['baseline'], minNPixel=config['minNPixel'])
        IntBin.WritePeakBinaryFile(
            snp, f"{config['binOut']}z{ilayer}_{str(frame_index).zfill(6)}.bin{idet}")

    return (f"[L{ilayer} D{idet}] reduced rotations {rstart}-{rstart + numFiles - 1}"
            f" | {time.perf_counter()-t0:.2f}s")

def image_reduction(config):
    '''
    Multiple median image reduction process
    Example config
    config = {
    'dety': 2048, #Detector size
    'detz': 2048, #Detector size
    'medOut': '/mnt/data/mzippere/test/test_', #Median image output directory
    'binOut': '/mnt/data/mzippere/test/test_', #Binarized image output directory
    'imgPrefix': '/mnt/data/mzippere/pokharel_jun25/nf/nf_Ti_1/nf_Ti_1_', #File path up to index for finding images
    'NDigit': 6, #Number of digits the index is zero padded to in the image file name
    'extension': '.tif', #Image file extension

    'imgFormat': 'tif', #'tif' for one file per image, 'h5' for one h5 image stack per detector
    'h5Path': 'exchange/data', #Dataset holding the (NRot, dety, detz) stack, autodetected if unset
    'h5StartFrame': 0, #Row of the first rotation image within the stack
    'h5Img': None, #Set only when a single archive holds every detector, overrides the file name above

    'NMedian': 2, #Number of medians per detector
    'blanket': 5, #Flat subtraction
    'LoGsig': 1.5, #Laplacian-of-Gaussian broadness
    'LoGcut': -1e-3, #Laplacian-of-Gaussian threshold
    'baseline': 0, #Minimum intensity for peak extraction
    'minNPixel': 4, #Minimum number of pixels for peak extraction

    'layerIdx': [0], #Layers to process, must be a list
    'NDet': 2, #Number of detectors
    'NRot': 20, #Images per detector to use, images recorded beyond this are ignored
    'startIdx': 44033, #File start index

    'fixLast': True, #Whether the last median absorbs the remainder when NRot/NMedian is not whole
    'verbose': True, #Defaults to True if left unset, prints out extra information on where the code is

    'NWorker': 56, #Processes to reduce frames on, defaults to 40
    'NFramePerTask': 8, #Frames per reduction task, defaults to 8
}
    Each detector is done in two passes.  The first computes the NMedian median
    backgrounds, one per process, each holding its own block of images and
    releasing it once the median is saved to medOut.  The second rereads the
    images and spreads subtraction, filtering, extraction and writing over
    NWorker processes in runs of NFramePerTask frames, holding a single image per
    process.  Peak memory is therefore set by the first pass, at NMedian blocks
    of NRot/NMedian images, rather than by the whole detector at once.

    Binary output is always numbered z{layer}_{rot:06d}.bin{det} with rot running
    0 to NRot-1, while NDigit describes the padding of the input file names only.

    With imgFormat 'h5' each recorded file is a whole rotation stack rather than a
    single image, so startIdx is the number of the first archive and the file for a
    given detector is imgPrefix + (startIdx + layer*NDet + det) + extension, e.g.
    NF_Au_cube_0802_0708.h5 and NF_Au_cube_0802_0709.h5 for a two detector scan
    with imgPrefix 'NF_Au_cube_0802_', NDigit 4, extension '.h5', startIdx 708.
    Rotation rot of that detector is read from row h5StartFrame + rot.
    '''
    if isinstance(config, str):
        with open(config,'r') as f:
            config = yaml.safe_load(f)
    elif not isinstance(config, dict):
        print(f'Input config must be dictionary or path to yaml file')
        return

    # main loop
    print(50*' ', datetime.now().strftime("%H:%M:%S"))
    FilePerMedian = int(config['NRot']/config['NMedian'])
    NWorker = int(config.get('NWorker') or 40)
    NFramePerTask = int(config.get('NFramePerTask') or 8)

    # rotations covered by each median, the last one taking the remainder
    lEdge = [imed*FilePerMedian for imed in range(config['NMedian'])]
    lEdge.append(config['NRot'] if config['fixLast'] else config['NMedian']*FilePerMedian)
    lBlock = [(lEdge[i], lEdge[i + 1] - lEdge[i]) for i in range(config['NMedian'])]

    for ilayer in config['layerIdx']:
        print('Layer number ', ilayer)
        startlayert = time.perf_counter()
        print('Detector ', config['NDet'])

        for idet in range(config['NDet']):
            lMed = [f"{config['medOut']}layer{ilayer}_det{idet}_med{imed}.npy"
                    for imed in range(config['NMedian'])]

            # Pass 1: one median per worker, each holding its own block of images
            with ProcessPoolExecutor(max_workers=min(config['NMedian'], NWorker)) as executor:
                for task in [executor.submit(compute_median_block, ilayer, idet, rs, nf, lMed[i], config)
                             for i, (rs, nf) in enumerate(lBlock)]:
                    print(task.result())

            # Pass 2: frames spread over the whole pool, one image per worker
            with ProcessPoolExecutor(max_workers=NWorker) as executor:
                for task in [executor.submit(reduce_frame_chunk, ilayer, idet, cstart,
                                             min(NFramePerTask, rs + nf - cstart), lMed[i], config)
                             for i, (rs, nf) in enumerate(lBlock)
                             for cstart in range(rs, rs + nf, NFramePerTask)]:
                    print(task.result())

        print(10*' ', f'Finished layer {ilayer} in {time.perf_counter() - startlayert:.2f} sec')
    return

def median_background(initial,startIdx,outInitial, NRot=720, NDet=2,NLayer=1,layerIdx=[0],digitLength=6,end='.tif', imgshape=[2048,2048],logfile=None):
    '''
    take median over omega as background
    initial: finle name initial. e.g.:'/home/heliu/work/shahani_feb19_part/nf_part/dummy_2_rt_before_heat_nf/dummy_2_rt_before_heat_nf_'
    startIdx: idex of first imagej.
    outInitial: output initial
    NRot: number of omega interval.
    NDet: number of detector.
    NLayer: number of layer.
    layerIdx: the index used for layers
    digitLength: length of digit in file name.
    end: file end format.
    imgshape: image resolution
    '''
    lBkg = []
    imgStack = np.empty([imgshape[0], imgshape[1], NRot],dtype=np.int32)
    start = time.time()
    if len(layerIdx) != NLayer:
        raise ValueError('layer index must be lenghth of NLayer')
    for layer in range(NLayer):
        print(f'layer: {layer}')
        if logfile is not None:
            logfile.write('')
        for det in range(NDet):
            print(f'det: {det}')
            for rot in range(NRot):
                print(f'rot: {rot}')
                idx = layer * NDet * NRot + det * NRot + rot + startIdx
                fName = f'{initial}{str(idx).zfill(digitLength)}{end}'
                print(fName)
                if logfile is not None:
                    logfile.write(f'layer: {layerIdx[layer]}, det: {det}, rot: {rot}, {fName} \n')
                try:
                    imgStack[:,:,rot] = tiff.imread(fName)
                    print('img loaded')
                except FileNotFoundError:
                    print(f'FILE NOT FOUND!!! {fName}')
                    if logfile is not None:
                        logfile.write(f'FILE NOT FOUND!!! {fName} \n ')
                except IndexError:
                    imgStack[:,:,rot] = np.zeros([2048,2048])
                    if logfile is not None:
                        logfile.write(f'FILE DESTROYED!!! {fName} \n ')
                #sys.stdout.write(f'\r {rot}')
                #sys.stdout.flush()
            if logfile is not None:
                logfile.write(f'applying median filter \n')
            bkg = np.median(imgStack, axis=2)
            if logfile is not None:
                logfile.write(f'complete median filter \n')
            try:
                np.save(f'{outInitial}_z{layerIdx[layer]}_det_{det}.npy', bkg)
                tiff.imwrite(f'{outInitial}_z{layerIdx[layer]}_det_{det}.tiff', bkg.astype(np.int32))
                print(f'saved bkg as {outInitial}_z{layer}_det_{det}.tiff/npy \n')
                if logfile is not None:
                    logfile.write(f'saved bkg as {outInitial}_z{layerIdx[layer]}_det_{det}.tiff/npy \n')
            except:
                print(f'FAIL SAVING as {outInitial}_z{layer}_det_{det}.tiff/npy \n')
                if logfile is not None:
                    logfile.write(f'FAIL SAVING as {outInitial}_z{layerIdx[layer]}_det_{det}.tiff/npy \n')
                    
            lBkg.append(bkg)
    end = time.time()
    print('\r')
    print(end - start)
    return lBkg

@jit(nopython=True,parallel=True)
def extract_peak(label,N, imgSubMed,imgSub, minNPixel, baseline):
    '''
    extract peaks out,
    : N:
        N is not used anymore, since label will be dilated and N will change.
    0.0079seconds
    '''
    lXOut = [0]
    lYOut = [0]
    lIDOut = [0] # make sure there will be output.
    lIntensityOut = [0]
    lXTmp = []
    lYTmp = []
    lIDTmp = []
    lSubMedTmp = []
    lSubTmp = []
    lIdxStart = [] # (begin,end,begin,end...), index of lists belongs to different label value.
    visited = label==0
    lXOut.append(1)
    lYOut.append(1)
    lIDOut.append(1)
    lIntensityOut.append(1)
    NX = label.shape[0]
    NY = label.shape[1]
    idx = 0
    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            # flood fill 
            if not visited[x,y]:
                lXTmp.append(x)
                lYTmp.append(y)
                lIDTmp.append(label[x,y])
                lSubMedTmp.append(imgSubMed[x,y])
                lSubTmp.append(imgSub[x,y])
                queX = [x]
                queY = [y]
                lIdxStart.append(idx)
                idx += 1
                while queX:
                    for dx in [-1,0,1]:
                        for dy in [-1,0,1]:
                            newX = queX[0]+dx
                            newY = queY[0]+dy
                            if newX>=0 and newX<NX and newY>=0 and newY<NY:
                                if not visited[newX, newY] and label[newX, newY]==label[queX[0],queY[0]]:
                                    queX.append(newX)
                                    queY.append(newY)
                                    lXTmp.append(newX)
                                    lYTmp.append(newY)
                                    lIDTmp.append(label[newX, newY])
                                    lSubMedTmp.append(imgSubMed[newX,newY])
                                    lSubTmp.append(imgSub[newX,newY])
                                    visited[newX, newY] = 1
                                    idx +=1
                    queX.pop(0)
                    queY.pop(0)
                lIdxStart.append(idx)
    #print(len(lIdxStart), N) # todo: why they can mismatch????
    
    for i in range(int(len(lIdxStart)/2)):
        start = lIdxStart[2*i]
        end = lIdxStart[2*i + 1]
        vMax = np.max(np.array(lSubMedTmp[start:end]))
        #print(vMax)
        if vMax > baseline:
            lXX = []
            lYY = []
            lVV = []
            lIDTmp = []
            for j in range(start, end):
                if lSubTmp[j]>(max(vMax*0.1,1)): # if subtraction image above 10% of max intensity, take it as part of the peak.
                    lXX.append(lXTmp[j])
                    lYY.append(lYTmp[j])
                    lVV.append(lSubTmp[j])
                    lIDTmp.append(i)
            if len(lXX)>minNPixel:
                lXOut.extend(lXX)
                lYOut.extend(lYY)
                lIntensityOut.extend(lVV)
                lIDOut.extend(lIDTmp)
    #print('..')
    #print(lXOut, lYOut, lIDOut, lIntensityOut)
    return lXOut, lYOut, lIDOut, lIntensityOut

def segmentation_numba(img, bkg, baseline=10, minNPixel=4,medianSize=3):
    '''
    return x,y,intensity,id
    '''
    #start = time.time()
    imgSub = img- bkg
    imgSubMed = ndi.median_filter(imgSub,size=medianSize)
    imgBase = imgSubMed - baseline
    imgBase[imgBase<0] = 0
    imgBaseMedian = ndi.median_filter(imgBase, size=medianSize)
    imgBaseMedian = imgBaseMedian.astype(np.uint16)
    gaussian = cv2.GaussianBlur(imgBaseMedian,ksize=(0,0),sigmaX=1.5,sigmaY=1.5,borderType=0)
    log = cv2.Laplacian(gaussian,cv2.CV_64F)
    #log = ndi.gaussian_laplace(imgBaseMedian,sigma=1.5)
    fillHole = ndi.binary_fill_holes(log<0)
    label,N = ndi.label(fillHole)
    #label = ndi.grey_dilation(label,size=(3,3))
    #start = time.time()
    lX, lY, lID, lIntensity = extract_peak(label,N, imgSubMed,imgSub, minNPixel, baseline)
    #end = time.time()
    #print(f'time taken:{end- start}')
    return (img.shape[1]- 1 - np.array(lY)).astype(np.int32), np.array(lX).astype(np.int32), np.array(lIntensity).astype(np.int32), np.array(lID).astype(np.int32)


def segmentation(img, bkg, baseline=10, minNPixel=4, medianSize=3):
    '''
    return x,y,intensity,id
    '''
    start = time.time()
    imgSub = img- bkg
    imgSubMed = ndi.median_filter(imgSub,size=medianSize)
    imgBase = imgSubMed - baseline
    imgBase[imgBase<0] = 0
    imgBaseMedian = ndi.median_filter(imgBase, size=medianSize)
    log = ndi.gaussian_laplace(imgBaseMedian,sigma=1.5)
    label,N = ndi.label(log<0)
    label = ndi.grey_dilation(label,size=(9,9))
    lX = []
    lY = []
    lID = []
    lIntensity = []
    #start = time.time()
# fill hole??? probably not a good idea in some cases.
    for i in range(N):
        mask = (label==i)
        # fill hole??? probably not a good idea in some cases.
        #mask = ndi.binary_fill_holes(mask)
        #mask = ndi.binary_dilation(mask,iterations=2)
        vMax = np.max(imgSubMed[mask].ravel())
        if vMax > baseline:
            x, y = np.where(mask*(imgSub>max(vMax*0.1,1)))
            if x.size>minNPixel:
                for i,xx in enumerate(x):
                    lX.append(xx)
                    yy = y[i]
                    lY.append(yy)
                    lIntensity.append(imgSub[xx,yy])
                    lID.append(label[xx,yy])
    end = time.time()
    print(f'time taken:{end- start}')
    return (img.shape[1]- 1 - np.array(lY)).astype(np.int32), np.array(lX).astype(np.int32), np.array(lIntensity).astype(np.int32), np.array(lID).astype(np.int32)

def reduce_image(initial,startIdx,bkgInitial,binInitial, NRot=720, NDet=2,NLayer=1,idxLayer=[0],digitLength=6,end='.tif', imgshape=[2048,2048],
                baseline=10, minNPixel=4):
    '''
    example usage:
        # reduce a layer of images together:
        import IntBin
        import time
        startIdx = 180904
        NRot = 720
        NDet = 1
        NLayer = 1
        end = '.tif'
        initial = '/home/heliu/work/shahani_feb19_part/nf_part/dummy_2_rt_before_heat_nf/dummy_2_rt_before_heat_nf_'
        bkgInitial = 'test_output_bkg'
        binInitial = 'dummy_2_rt_before_heat_nf_test_'
        reduce_image(initial,startIdx,bkgInitial,binInitial, NRot=NRot, NDet=Det,NLayer=1,idxLayer=[0],digitLength=6,end='.tif', imgshape=[2048,2048], baseline=10, minNPixel=4)        
    '''
    lBkg = median_background(initial, startIdx, bkgInitial,NRot=NRot, NDet=NDet, NLayer=NLayer,end=end)
    print('bkg images created')
    idxBkg = 0
    idxLayer = [0]
    for layer in range(NLayer):
        for det in range(NDet):
            bkg = lBkg[idxBkg]
            idxBkg += 1
            for rot in range(NRot):
                idx = layer * NDet * NRot + det * NRot + rot + startIdx
                fName = f'{initial}{str(idx).zfill(digitLength)}{end}'
                img = tiff.imread(fName)
                binFileName = f'{binInitial}z{idxLayer[layer]}_{rot}.bin{det}'
                print(binFileName)
                snp = segmentation(img, bkg, baseline=baseline, minNPixel=minNPixel)
                IntBin.WritePeakBinaryFile(snp, binFileName)

def integrate_tiff(tiffInitial, startIdx, digit, extention, NImage, NInt,outInitial, outStartIdx):
    '''
    startIdx = 333224
    tiffInitial = '/media/heliu/Seagate Backup Plus Drive/krause_jul19/nf/s1350_100_1_nf/s1350_100_1_nf_'
    digit = 6
    extention = 'tif'
    NInt = 4  # integrate 4 images into 1.
    NImage = 1440*22 # number of images before integration
    outInitial = '/media/heliu/Seagate Backup Plus Drive/krause_jul19/nf/s1350_100_1_nf/s1350_100_1_nf_int4_'
    outStartIdx = 0 # starting index of output image
    '''
    for i in range(NImage):
        fName = f'{tiffInitial}{(i+startIdx):0{digit}d}{extention}'
        if not os.path.exists(fName):
            raise FileExistsError(f'file not found: {fName}')
    lTiff = []
    for i in range(NImage):
        fName = f'{tiffInitial}{(i+startIdx):0{digit}d}{extention}'
        sys.stdout.write(f'\r {i}/{NImage}: {fName}')
        sys.stdout.flush()
        if i%NInt ==0:
            lTiff = []
            lTiff.append(tiff.imread(fName))
        else:
            lTiff.append(tiff.imread(fName))
        if len(lTiff) == NInt:
            outImg = np.sum(np.array(lTiff),axis=0).astype(np.int32)
            #print(outImg.shape)
            tiff.imwrite(f'{outInitial}{(i//NInt+outStartIdx):0{digit}d}{extention}', outImg)
            sys.stdout.write(f'\r writing: {outInitial}{(i//NInt+outStartIdx):0{digit}d}{extention}')
            sys.stdout.flush()
if  __name__ == '__main__':
    
    import sys
    sys.path.insert(0, '/home/heliu/work/dev/HEXOMAP/')
    import IntBin
    plt.rcParams["figure.figsize"] = (10,10)

    # images
    startIdx = 180904
    NRot = 720
    NDet = 1
    NLayer = 1
    end = '.tif'
    initial = '/home/heliu/work/shahani_feb19_part/nf_part/dummy_2_rt_before_heat_nf/dummy_2_rt_before_heat_nf_'
    fName = f'{initial}{startIdx:06d}{end}'
    img = tiff.imread(fName)
    bkg = np.load('test_output_bkg_z0_det_0.npy')
    lX, lY, lIntensity, lID = segmentation(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)