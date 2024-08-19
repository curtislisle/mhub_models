"""
-------------------------------------------------
MHub - run the NCI RMS segmentation pipeline
-------------------------------------------------

-------------------------------------------------
---------------------------------------------------
Author: Curtis Lisle
Email:  clisle@knowledgevis.com
based on examples from Dennis Bontempi
model developed and trained by Dr. Hyun Jung, NCI

This model outputs a fractional segmentation dicom object with four classes:
(ARMS,ERMS,STROMA,NECROSIS). 
---------------------------------------------------
"""

import os, subprocess, shutil
from mhubio.core import Instance, InstanceData, IO
from mhubio.modules.runner.ModelRunner import ModelRunner

# declarations needed for inline mmodel execution

import os
import numpy as np
import time
import torch
from collections import OrderedDict
import json
import large_image
import large_image_source_dicom

import sys
import random
#import argparse
import torch.nn as nn
#import cv2

import glob
from skimage.io import imread, imsave
from skimage import filters
from skimage.color import rgb2gray
import gc

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import albumentations as albu
import segmentation_models_pytorch as smp


#------ Start Global definitiona ---------------------

# define global variable that is set according to whether GPUs are discovered
if torch.cuda.is_available():
    USE_GPU = True
    print('GPU is available')
else:
    USE_GPU = False
    print('GPU is not available. Using CPU')


ml = nn.Softmax(dim=1)

# used for colorization of output image
NE = 50
ST = 100
ER = 150
AR = 200

# how often to update status
PRINT_FREQ = 20
# default batch size, at which model was validated
BATCH_SIZE = 80

ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DEVICE = 'cuda'

# the weights file is in the same directory, so make this path reflect that.  If this is 
# running in a docker container, then we should assume the weights are at the toplevel 
# directory instead


# these aren't used in the girder version, no files are directly written out 
# by the routines written by FNLCR (Hyun Jung)
WSI_PATH = '.'
PREDICTION_PATH = '.'

IMAGE_SIZE = 384
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
REGION_RGBA = 4
CHANNELS = 3
NUM_CLASSES = 5
CLASS_VALUES = [0, 50, 100, 150, 200]

BLUE = [0, 0, 255] # ARMS: 200
RED = [255, 0, 0] # ERMS: 150
GREEN = [0, 255, 0] # STROMA: 100
YELLOW = [255, 255, 0] # NECROSIS: 50
EPSILON = 1e-6

# what magnification should this pipeline run at
ANALYSIS_MAGNIFICATION = 10.0
THRESHOLD_MAGNIFICATION = 2.5
ASSUMED_SOURCE_MAGNIFICATION = 39.5882818685669

# what % interval we should print out progress so it can be snooped by the web interface
REPORTING_INTERVAL = 20

rot90 = albu.Rotate(limit=(90, 90), always_apply=True)
rotn90 = albu.Rotate(limit=(-90, -90), always_apply=True)

rot180 = albu.Rotate(limit=(180, 180), always_apply=True)
rotn180 = albu.Rotate(limit=(-180, -180), always_apply=True)

rot270 = albu.Rotate(limit=(270, 270), always_apply=True)
rotn270 = albu.Rotate(limit=(-270, -270), always_apply=True)

hflip = albu.HorizontalFlip(always_apply=True)
vflip = albu.VerticalFlip(always_apply=True)
tpose = albu.Transpose(always_apply=True)

pad = albu.PadIfNeeded(p=1.0, min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=(255, 255, 255), mask_value=0)

#  This is based on the hidicom output example listed in the readthedocs documentation
CHANNEL_DESCRIPTION = {}
CHANNEL_DESCRIPTION['chan_0'] = 'Background'
CHANNEL_DESCRIPTION['chan_1'] = 'Necrosis'
CHANNEL_DESCRIPTION['chan_2'] = 'Stroma'
CHANNEL_DESCRIPTION['chan_3'] = 'ARMS'
CHANNEL_DESCRIPTION['chan_4'] = 'ERMS'
CHANNEL_DESCRIPTION['chan_1_prob'] = 'Necrosis_prob'
CHANNEL_DESCRIPTION['chan_2_prob'] = 'Stroma_prob'
CHANNEL_DESCRIPTION['chan_3_prob'] = 'ARMS_prob'
CHANNEL_DESCRIPTION['chan_4_prob'] = 'ERMS_prob'

# Dictionary mapping text label found in the XML annoations to tuple of
# (finding_type, finding_category) codes to encode that finding

# SNOMED codes for the RMS model class outputs

import highdicom as hd

finding_codes = {
    "STROMA": (
        hd.sr.CodedConcept(
            meaning="Connective tissue",
            value="181769001",
            scheme_designator="SCT",
        ),
        hd.sr.CodedConcept(
            meaning="Body substance",
            value="91720002",
            scheme_designator="SCT",
        ),
    ),
    "ARMS": (
        hd.sr.CodedConcept(
            meaning="Alveolar rhabdomyosarcoma",
            value="63449009",
            scheme_designator="SCT",
        ),
        hd.sr.CodedConcept(
            meaning="Morphologic abnormality",
            value="49755003",
            scheme_designator="SCT",
        ),
    ),
    "ERMS": (
        hd.sr.CodedConcept(
            meaning="Embryonal rhabdomyosarcoma",
            value="14269005",
            scheme_designator="SCT",
        ),
        hd.sr.CodedConcept(
            meaning="Morphologic abnormality",
            value="49755003",
            scheme_designator="SCT",
        ),
    ),
    "NECROSIS": (
        hd.sr.CodedConcept(
            meaning="Necrosis",
            value="6574001",
            scheme_designator="SCT",
        ),
        hd.sr.CodedConcept(
            meaning="Morphologic abnormality",
            value="49755003",
            scheme_designator="SCT",
        ),
    ),
}

#------ End Global definitions ---------------------


#  using highdicom library (from MGH) for dicom support. see examples at:
#  https://highdicom.readthedocs.io/en/latest/usage.html


#@IO.Config('batchsize', int, 64, the='Number of slices to be processed simultaneously. A smaller batch size requires less memory but may be slower.')
#@IO.Config('fractional', bool, False, the='output a fractional segmentation mask indicating probability of class membership. Default is Binary Segmentation')
class Patho_RMS_Runner_Fractional(ModelRunner):
    #fractional : bool
    #batchsize: int

    # Question:  I don't understand how the channels are specified in the 'roi' argument
    @IO.Instance()
    @IO.Input('image', 'dicom:mod=sm',  the='input whole slide image')
    @IO.Output('structures', 'pathology_rms.seg.dcm', 'dicomseg:mod=seg:model=patho_rms', bundle='model', the='predicted tissue classes')
    def task(self, instance: Instance, image: InstanceData, structures: InstanceData) -> None:
        self.log.debug("Running the segmentation.")
        print('fractional segmentation mode is enabled')

        # *** hardcode the model weights location until figuring out
        # the initialization method
        self.log.debug('Need to pull weights from public repository!')
        #modelCheckpointFilePath = '/home/clisle/proj/slicer/PW39/rms-infer-code-standalone/'
        modelCheckpointFilePath = '/root/.cache/torch/hub/checkpoints/'

        # the input image passed is the containing directory, not a file, so look up a file
        inputImagePath = self.findDicomWsiFile(image.abspath)
        self.log.debug(f'discovered input file is {inputImagePath}')
        self.log.debug(f'output path is {structures.abspath}')
        # run model. Should this be in a subprocess?
        outfile = self.infer_rhabdo(modelCheckpointFilePath,inputImagePath,structures.abspath)
       

    # look in the directory for a dicom file
    def findDicomWsiFile(self, dirPath):
        file_list = []
        for root, dirs, files in os.walk(dirPath):
            for file in files:
                file_list.append(os.path.join(root, file))
        # *** this is a hack, need to find the right file
        return file_list[0]


    def infer_rhabdo(self,modelCheckpointFilePath,image_file,out_file,**kwargs):
        self.log.debug(" input image filename = {}".format(image_file))
        # setup the GPU environment for pytorch
        if USE_GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            DEVICE = 'cuda'
            self.log.debug('using GPU')
        else:
            DEVICE = 'cpu'
            self.log.debug('using CPU')

        self.log.debug('perform forward inferencing')
        start_inference_mainthread(modelCheckpointFilePath,image_file,out_file)
        self.log.debug('inferencing complete')

        # return the name of the output file
        return out_file


# supporting subroutines
#-----------------------------------------------------------------------------

def _infer_batch(model, test_patch):
    # print('Test Patch Shape: ', test_patch.shape)
    with torch.no_grad():
        logits_all = model(test_patch[:, :, :, :])
        logits = logits_all[:, 0:NUM_CLASSES, :, :]
    prob_classes_int = ml(logits)
    prob_classes_all = prob_classes_int.cpu().numpy().transpose(0, 2, 3, 1)

    return prob_classes_all

def _augment(index, image):

    if index == 0:
        image= image

    if index == 1:
        augmented = rot90(image=image)
        image = augmented['image']

    if index ==2:
        augmented = rot180(image=image)
        image= augmented['image']

    if index == 3:
        augmented = rot270(image=image)
        image = augmented['image']

    if index == 4:
        augmented = vflip(image=image)
        image = augmented['image']

    if index == 5:
        augmented = hflip(image=image)
        image = augmented['image']

    if index == 6:
        augmented = tpose(image=image)
        image = augmented['image']

    return image
    
def _unaugment(index, image):

    if index == 0:
        image= image

    if index == 1:
        augmented = rotn90(image=image)
        image = augmented['image']

    if index ==2:
        augmented = rotn180(image=image)
        image= augmented['image']

    if index == 3:
        augmented = rotn270(image=image)
        image = augmented['image']

    if index == 4:
        augmented = vflip(image=image)
        image = augmented['image']

    if index == 5:
        augmented = hflip(image=image)
        image = augmented['image']

    if index == 6:
        augmented = tpose(image=image)
        image = augmented['image']

    return image


# return a string identifier of the basename of the current image file
def returnIdentifierFromImagePath(impath):
    # get the full name of the image
    file = os.path.basename(impath)
    # strip off the extension
    base = file.split('.')[0]
    return(base)


def isNotANumber(variable):
    # this try clause will work for integers and float values, since floats can be cast.  If the
    # variable is any other type (include None), the clause will cause an exception and we will return False
    try:
        tmp = int(variable)
        return False
    except:
        return True

# debug routine for printing strange sized tiles returned from patch etraction
def displayTileMetadata(tile,region, i, j):
    print('-----------------------------')
    print('wierd tile shape encountered:')
    print('Tile shape:',tile.shape)
    print('Region:',region)
    print('i:',i,'j:',j)


# turn a partial tile (smaller than the model size) into a full tile by padding with slide
# background RGBA = (240,240,240,240). We accomplish this by creating a full tile and 
# copying the actual slide subset data into the full tile before returning

def fillPartialTile(partialTile):
    sizeOfX = partialTile.shape[0]
    sizeOfY = partialTile.shape[1]
    fullTile = np.ones((IMAGE_SIZE, IMAGE_SIZE, REGION_RGBA),np.uint8)*240
    # this is is on the corner, we need to copy over a partial x and y record
    if sizeOfX < IMAGE_WIDTH and sizeOfY < IMAGE_HEIGHT:
        fullTile[:sizeOfX,:sizeOfY,:] = partialTile[:sizeOfX,:sizeOfY,:]
    # we are at the end of a row, so we need to copy over a partial x record
    elif sizeOfX < IMAGE_WIDTH:
        fullTile[:sizeOfX,:,:] = partialTile[:sizeOfX,:,:]
    # we are on the bottom line, so we need to copy over a partial y record
    elif sizeOfY < IMAGE_HEIGHT:
        fullTile[:,:sizeOfY,:] = partialTile[:,:sizeOfY,:]
    return fullTile
    

#---------------- main inferencing routine ------------------
def _inference(model, image_path, BATCH_SIZE, num_classes, kernel, num_tta=1):

    model.eval()

    # open an access handler on the large image
    #source = large_image.getTileSource(image_path)
    #source = large_image_source_tiff.open(image_path)
    source = large_image_source_dicom.open(image_path)

    # print image metadata
    metadata = source.getMetadata()
    print(metadata)
    print('sizeX:', metadata['sizeX'], 'sizeY:', metadata['sizeY'], 'levels:', metadata['levels'])

    # figure out the size of the actual image and the size that this analysis
    # processing will run at.  The size calculations are made in two steps to make sure the
    # rescaled threshold image size and the analysis image size match without rounding error

    height_org = metadata['sizeY']
    width_org = metadata['sizeX']

    # if we are processing using a reconstructed TIF from VIPS, there will not be a magnification value.
    # So we will assume a native magnification.  See the constants defined somewhere around line 127

    if isNotANumber(metadata['magnification']):
        print('warning: No magnfication value in source image. Assuming the source image is at ',
            ASSUMED_SOURCE_MAGNIFICATION,' magnification')
        metadata['magnification'] = ASSUMED_SOURCE_MAGNIFICATION
        assumedMagnification = True
    else:
        assumedMagnification = False
        # run at the exact magnifiction of the source and generate 25% size for OTSU
        ANALYSIS_MAGNIFICATION = metadata['magnification']
        THRESHOLD_MAGNIFICATION = ANALYSIS_MAGNIFICATION
        
    # the theoretical adjustment for the magnification would be as below:
    # height_proc = int(height_org * (ANALYSIS_MAGNIFICATION/metadata['magnification']))
    # width_proc = int(width_org * (ANALYSIS_MAGNIFICATION/metadata['magnification']))

    height_proc = int(height_org * THRESHOLD_MAGNIFICATION/metadata['magnification'])*int(ANALYSIS_MAGNIFICATION/THRESHOLD_MAGNIFICATION)
    width_proc = int(width_org * THRESHOLD_MAGNIFICATION/metadata['magnification'])*int(ANALYSIS_MAGNIFICATION/THRESHOLD_MAGNIFICATION)
    print('analysis image size :',height_proc, width_proc)

    basename_string = os.path.splitext(os.path.basename(image_path))[0]
    print('Basename String: ', basename_string)

    # generate a binary mask for the image
    height_otsu = int(height_proc * THRESHOLD_MAGNIFICATION/ANALYSIS_MAGNIFICATION)
    width_otsu = int(width_proc * THRESHOLD_MAGNIFICATION / ANALYSIS_MAGNIFICATION)
    print('size of threshold mask:',height_otsu,width_otsu)
    # this will always generate a 10x region size, even if the source image has lower resolution
    myRegion = {'top': 0, 'left': 0, 'width': width_org, 'height': height_org}


    if assumedMagnification:
        # we have to manage the downsizing to the threshold magnification.
        threshold_source_image, mimetype = source.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                                                            region=myRegion,output={'maxWidth':width_otsu,'maxHeight':height_otsu})
        print('used maxOutput for threshold size')
    else:

        threshold_source_image, mimetype = source.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                                                        region=myRegion,
                                                        scale={'magnification': THRESHOLD_MAGNIFICATION})

    print('OTSU image')
    print(threshold_source_image.shape)

    # strip off any extra alpha channel
    threshold_source_image = threshold_source_image[:,:,0:3]
    print(threshold_source_image.shape)
    thumbnail_gray = rgb2gray(threshold_source_image)
    val = filters.threshold_otsu(thumbnail_gray)
    # create empty output for threshold
    otsu_seg = np.zeros((threshold_source_image.shape[0], threshold_source_image.shape[1]), np.uint8)
    # generate a mask=true image where the source pixels were darker than the
    # # threshold value (indicating tissue instead of bright background)
    otsu_seg[thumbnail_gray <= val] = 255
    # OTSU algo. was applied at reduced scale, so scale image back up
    aug = albu.Resize(p=1.0, height=height_proc, width=width_proc)
    augmented = aug(image=otsu_seg, mask=otsu_seg)
    otsu_org = augmented['mask'] // 255
    print('rescaled threshold shape is:', otsu_org.shape)
    #imsave('otsu.png', (augmented['mask'] .astype('uint8')))
    print('Otsu segmentation finished')


    # initialize the output probability map
    prob_map_seg_stack = np.zeros((height_proc, width_proc, num_classes), dtype=np.float32)

    for b in range(num_tta):

        height = height_proc
        width = width_proc

        PATCH_OFFSET = IMAGE_SIZE // 2
        SLIDE_OFFSET = IMAGE_SIZE // 2
        print('using', (PATCH_OFFSET//IMAGE_SIZE*100),'% patch overlap')

        # these are the counts in the x and y direction.  i.e. how many samples across the image.
        # the divident is slide_offset because this is how much the window is moved each time
        heights = (height + PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET +1
        widths = (width + PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET +1
        print('heights,widths:',heights,widths)

        heights_v2 = (height + PATCH_OFFSET * 2) // (SLIDE_OFFSET)
        widths_v2 = (width + PATCH_OFFSET * 2) // (SLIDE_OFFSET)
        print('heights_v2,widths_v2',heights_v2,widths_v2)

        # extend the size to allow for the whole actual image to be processed without actual
        # pixels being at a tile boundary.

        # doubled to *4 and *8 when changed 408 #409 to //4
        height_ext = SLIDE_OFFSET * heights + PATCH_OFFSET * 2
        width_ext = SLIDE_OFFSET * widths + PATCH_OFFSET * 4
        print('height_ext,width_ext:',height_ext,width_ext)

        org_slide_ext = np.ones((height_ext, width_ext, 3), np.uint8) * 255
        otsu_ext = np.zeros((height_ext, width_ext), np.uint8)
        prob_map_seg = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)
        weight_sum = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)

        #org_slide_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, 0:3] = image_working[:, :,
        #                                                                                             0:3]

        # load the otsu results
        otsu_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width] = otsu_org[:, :]

        linedup_predictions = np.zeros((heights * widths, IMAGE_SIZE, IMAGE_SIZE, num_classes), dtype=np.float32)
        linedup_predictions[:, :, :, 0] = 1.0

        test_patch_tensor = torch.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=torch.float)
        if USE_GPU:
            test_path_tensor = test_patch_tensor.cuda(non_blocking=True)
        
        # get an identifier for the patch files to be written out as debugging
        unique_identifier = returnIdentifierFromImagePath(image_path)

        # decide how long this will take and prepare to give status updates in the log file
        iteration_count = heights*widths
        report_interval = iteration_count / (100 / REPORTING_INTERVAL)
        report_count = 0
        # report current state 
        percent_complete = 0

        patch_iter = 0
        inference_index = []
        position = 0
        stopcounter = 0

        for i in range(heights):
            for j in range(widths):
                #test_patch = org_slide_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                #             j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, 0:3]

                # specify the region to extract and pull it at the proper magnification.  If a region is outside
                # of the image boundary, the returned tile will be padded with white pixels (background).  The region
                # coordinates are in the coordinate frame of the original, full-resolution image, so we need to calculate
                # them from the analytical coordinates
                top_in_orig = int(i * SLIDE_OFFSET * metadata['magnification']/ANALYSIS_MAGNIFICATION)
                left_in_orig = int(j * SLIDE_OFFSET * metadata['magnification'] / ANALYSIS_MAGNIFICATION)
                image_size_in_orig = int(IMAGE_SIZE* metadata['magnification'] / ANALYSIS_MAGNIFICATION)
                myRegion = {'top': top_in_orig, 'left': left_in_orig, 'width': image_size_in_orig, 'height': image_size_in_orig}
                rawtile, mimetype = source.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                                                        region=myRegion, scale={'magnification': ANALYSIS_MAGNIFICATION},
                                                        fill="white",output={'maxWidth':IMAGE_SIZE,'maxHeight':IMAGE_SIZE})
                
                # if this is a boundary tile, then fill in the partial tile that comes from the getRegion call into 
                # a complete tile by adding a white-ish boundary to make the tile the usual, full size
                if (rawtile.shape[0] < IMAGE_SIZE) or (rawtile.shape[1] < IMAGE_SIZE):
                    #print('ran off the X or Y edge: xcorner:',xcorner,'ycorner:',ycorner,'shape:',tile.shape)
                    rawtile = fillPartialTile(rawtile)

                # strip off any extra channels, RGB only
                test_patch = rawtile[:,:,0:3]
                # print out funny shaped patches... 
                if (test_patch.shape[0] != IMAGE_SIZE) or (test_patch.shape[1] != IMAGE_SIZE):
                    displayTileMetadata(test_patch,myRegion,i,j)
                    print(test_patch.shape)
        
                otsu_patch = otsu_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE]
                if np.sum(otsu_patch) > int(0.05 * IMAGE_SIZE * IMAGE_SIZE):
                    inference_index.append(patch_iter)
                    test_patch_tensor[position, :, :, :] = torch.from_numpy(test_patch.transpose(2, 0, 1)
                                                                            .astype('float32') / 255.0)
                    position += 1
                patch_iter += 1

                if position == BATCH_SIZE:
                    batch_predictions = _infer_batch(model, test_patch_tensor)

                    for k in range(BATCH_SIZE):
                        linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

                    position = 0
                    inference_index = []

                # check that it is time to report progress.  If so, print it and flush I/O to make sure it comes 
                # out right after it is printed 
                report_count += 1
                if (report_count > report_interval):
                    percent_complete += REPORTING_INTERVAL
                    print(f'progress: {percent_complete}')
                    sys.stdout.flush()
                    report_count = 0


        # Very last part of the region.  This is if there is a partial batch of tiles left at the
        # end of the image.
        batch_predictions = _infer_batch(model, test_patch_tensor)
        for k in range(position):
            linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

        # finished with the model, clear the memory and GPU
        del test_patch_tensor
        del model
        if USE_GPU:
            torch.cuda.empty_cache()

        print('Inferencing complete. Constructing out image from patches')

        patch_iter = 0
        for i in range(heights):
            for j in range(widths):
                prob_map_seg[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE,:] \
                    += np.multiply(linedup_predictions[patch_iter, :, :, :], kernel)
                weight_sum[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE,:] \
                    += kernel
                patch_iter += 1
        #np.save("prob_map_seg.npy",prob_map_seg)
        #np.save('weight_sum.npy',weight_sum)
        print('Do not worry about the following divide by zero. It happens in valid output images')
        prob_map_seg = np.true_divide(prob_map_seg, weight_sum)
        
        # output the gaussian smoother to look at the overlaps
        #small_weights = (weight_sum[0:1000,0:1000,0:3]*255).astype(np.uint8)
        #np.save('gaussian_weights.np',small_weights)
        #cv2.imwrite("gaussian_weights.png", small_weights)


        # *********************************
        # this induced a 1/2 PATCH_OFFSET shift in the output image compared with the reference DICOM. 
        # so replace with a direct copy operation instead. 
        # *********************************
        #prob_map_valid = prob_map_seg[PATCH_OFFSET:PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, :]
        prob_map_valid = prob_map_seg[0:height, 0: width, :]

        # free main system memory since the images are big
        del prob_map_seg
        gc.collect()

        prob_map_valid = _unaugment(b, prob_map_valid)
        prob_map_seg_stack += prob_map_valid / num_tta

  
        # free main system memory since the images are big
        del prob_map_valid
        gc.collect()

    # save numpy in same directory as input image
    #fileNoExtension = os.path.basename(image_path).split('.')[0]
    #dirName = os.path.dirname(image_path)
    #numpyFileName = os.path.join(dirName,fileNoExtension+'_prob_map_seg_stack.npy')
    #np.save(numpyFileName, prob_map_seg_stack)
    print('returning probability map size:',prob_map_seg_stack.shape)
    return prob_map_seg_stack



def _gaussian_2d(num_classes, sigma, mu):
    x, y = np.meshgrid(np.linspace(-1, 1, IMAGE_SIZE), np.linspace(-1, 1, IMAGE_SIZE))
    d = np.sqrt(x * x + y * y)
    # sigma, mu = 1.0, 0.0
    k = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    k_min = np.amin(k)
    k_max = np.amax(k)

    k_normalized = (k - k_min) / (k_max - k_min)
    k_normalized[k_normalized<=EPSILON] = EPSILON

    kernels = [(k_normalized) for i in range(num_classes)]
    kernel = np.stack(kernels, axis=-1)

    print('Kernel shape: ', kernel.shape)
    print('Kernel Min value: ', np.amin(kernel))
    print('Kernel Max value: ', np.amax(kernel))

    return kernel


def reset_seed(seed):
    """
    ref: https://forums.fast.ai/t/accumulating-gradients/33219/28
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def load_best_model(model, path_to_model, best_prec1=0.0):
    if os.path.isfile(path_to_model):
        print("=> loading checkpoint '{}'".format(path_to_model))
        checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}), best_precision {}"
              .format(path_to_model, checkpoint['epoch'], best_prec1))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(path_to_model))


def inference_image(model, image_path, BATCH_SIZE, num_classes):
    kernel = _gaussian_2d(num_classes, 0.5, 0.0)
    prob_image = _inference(model, image_path, BATCH_SIZE, num_classes, kernel, 1)
    return prob_image


def start_inference_mainthread(modelWeightPath,image_file,out_file):
    reset_seed(1)

    best_prec1_valid = 0.
    torch.backends.cudnn.benchmark = True

    #saved_weights_list = sorted(glob.glob(WEIGHT_PATH + '*.tar'))
    #saved_weights_list = [os.path.join(modelWeightPath,'model_iou_0.7343_0.7175_epoch_60.pth.tar')] 
    saved_weights_list = [os.path.join(modelWeightPath,'rms_segment_fold_03.pth')]
    print(saved_weights_list)

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASS_VALUES),
        activation=ACTIVATION,
        aux_params=None,
    )

    model = nn.DataParallel(model)
    if USE_GPU:
        model = model.cuda()
    print('load pretrained weights')
    model = load_best_model(model, saved_weights_list[-1], best_prec1_valid)
    print('Loading model is finished!!!!!!!')
    # return image data so toplevel task can write it out
    prob_image = inference_image(model,image_file, BATCH_SIZE, len(CLASS_VALUES))
    # pass the original dicom file, so header information can be read.  
    print('writing fractional segmentation')
    writeDicomFractionalSegObject(image_file,prob_image,out_file)
   


#---------------- DICOM export --------


from pathlib import Path

import highdicom as hd
import dicomslide
from pydicom.sr.codedict import codes
from pydicom.filereader import dcmread
from pydicom import Dataset
from dicomweb_client import DICOMfileClient
from tempfile import TemporaryDirectory
from typing import Tuple

def disassemble_total_pixel_matrix(
    seg_total_pixel_matrix: np.ndarray,
    source_image_metadata: Dataset,
) -> np.ndarray:
    """Disassemble a total pixel matrix into individual tiles.

    Parameters
    ----------
    seg_total_pixel_matrix: numpy.ndarray
        Total pixel matrix of the segmentation as a 2D NumPy array.
    source_metadata: pydicom.Dataset
        DICOM metadata of the source image.

    Returns
    -------
    numpy.ndarray
        Stacked image tiles

    """
    if seg_total_pixel_matrix.ndim != 2:
        raise ValueError(
            "Total pixel matrix has unexpected number of dimensions."
        )

    # Need a client object to work with DICOM slide so just create a dummy one
    with TemporaryDirectory() as tmpdir:
        client = DICOMfileClient(f"file://{tmpdir}")

        im_tpm = dicomslide.TotalPixelMatrix(client, source_image_metadata)

        tile_rows, tile_cols, _ = im_tpm.tile_shape

        return dicomslide.disassemble_total_pixel_matrix(
            seg_total_pixel_matrix,
            im_tpm.tile_positions,
            tile_rows,
            tile_cols,
        )




def writeDicomFractionalSegObject(image_path, seg_image, out_path):

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path(image_path)

    # Read SM Image data set from PS3.10 files on disk.  This will provide the 
    # reference image size and other dicom header information
    image_dataset = dcmread(str(image_file))

    # function stolen from idc-pan-cancer-archive repository to re-tile the numpy to match the tiling
    # from the source image.  It only works for a 3D array, so we have to repeat for each channel and
    # stack the results.
 
    print('passing in a numpy array of shape:',seg_image.shape)
    mask_1 = disassemble_total_pixel_matrix(seg_image[:,:,1],image_dataset)
    mask_2 = disassemble_total_pixel_matrix(seg_image[:,:,2],image_dataset)
    mask_3 = disassemble_total_pixel_matrix(seg_image[:,:,3],image_dataset)
    mask_4 = disassemble_total_pixel_matrix(seg_image[:,:,4],image_dataset)
    print('disassembled dimensions:',mask_1.shape)
    mask = np.zeros((mask_1.shape[0],mask_1.shape[1], mask_1.shape[2],4), np.float32)
    mask[:,:,:,0] = mask_1
    mask[:,:,:,1] = mask_2
    mask[:,:,:,2] = mask_3
    mask[:,:,:,3] = mask_4
 
    # Describe the algorithm that created the segmentation
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='FNLCR_IVG_RMS_probability_iou_0.7343_epoch_60',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )
    # use the method from Chris Bridge to create the segment descriptions because the correct
    # SNOMED  codes are already setup in the metadata_config file.
    segment_descriptions = []
    
    for number, (label, (prop_code, cat_code)) in enumerate(
        finding_codes.items(),
        start=1
    ):
        desc = hd.seg.SegmentDescription(
            segment_number=number,
            segment_label=label,
            segmented_property_category=cat_code,
            segmented_property_type=prop_code,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification= algorithm_identification
        )
        segment_descriptions.append(desc)
   
    # Create the Segmentation instance
    seg_dataset = hd.seg.Segmentation(
        source_images=[image_dataset],
        pixel_array=mask,
        #tile_pixel_array=True,
        segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
        dimension_organization_type= 'TILED_FULL',
        omit_empty_frames=False,
        #segment_descriptions=[description_segment_1,description_segment_2,description_segment_3,description_segment_4],
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=3,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        # the following two entries are added because the output resolution is different from the source
        #pixel_measures=derived_pixel_measures,
        #plane_positions= derived_plane_positions,
        manufacturer='NCI/FNLCR',
        manufacturer_model_name='FNLCR_IVG_RMS_iou_0.7343_0.7175_epoch_60',
        software_versions='fractional_seg_mhub_v1',
        device_serial_number='Unknown'
    )
    seg_dataset.save_as(out_path)

