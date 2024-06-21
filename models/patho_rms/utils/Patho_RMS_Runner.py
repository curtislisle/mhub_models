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
USE_GPU = True

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
REPORTING_INTERVAL = 5

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

#------ End Global definitiona ---------------------


#  using highdicom library (from MGH) for dicom support. see examples at:
#  https://highdicom.readthedocs.io/en/latest/usage.html


#@IO.Config('batchsize', int, 64, the='Number of slices to be processed simultaneously. A smaller batch size requires less memory but may be slower.')
class Patho_RMS_Runner(ModelRunner):
    
    #batchsize: int

    # Question:  I don't understand how the channels are specified in the 'roi' argument
    @IO.Instance()
    @IO.Input('image', 'dicom:mod=path',  the='input whole slide image')
    @IO.Output('structures', 'pathology_rms.seg.dcm', 'dicom:mod=seg:model=patho_rms', bundle='model', the='predicted tissue classes')
    def task(self, instance: Instance, image: InstanceData, structures: InstanceData) -> None:
        # Question: is this just a logging output?
        self.v("Running the segmentation.")

        # *** hardcode the model weights location until figuring out
        # the initialization method
        print('Need to pull weights from public repository!')
        modelCheckpointFilePath = '/home/clisle/proj/slicer/PW39/rms-infer-code-standalone/'

        # run model. Should this be in a subprocess?
        outfile = self.infer_rhabdo(modelCheckpointFilePath,image.abspath,structures.abspath)


    def infer_rhabdo(self,modelCheckpointFilePath,image_file,out_file,**kwargs):
        print(" input image filename = {}".format(image_file))
        # setup the GPU environment for pytorch
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        DEVICE = 'cuda'

        print('perform forward inferencing')
        predict_image = start_inference_mainthread(modelCheckpointFilePath,image_file,out_file)
        print('inferencing complete')

        # generate unique names for multiple runs.  Add extension so it is easier to use. Put output
        # in the same directory as the output dicom file
        #out_dir = os.path.dirname(out_file)
        #outname_color = os.path.join(out_dir,image_file.split('.')[0]+'_predict.png')

        # write the output object using openCV  
        print('writing segmentation output')
        cv2.imwrite(out_file,predict_image)
        print('writing output completed')

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

def _gray_to_labelmap(input_probs):
    index_map = (np.argmax(input_probs, axis=-1)*50).astype('uint8')
    height = input_probs.shape[0]
    width = input_probs.shape[1]
    heatmap = np.zeros((height, width, 1), np.float32)
    # Background
    heatmap[index_map == 0, 0] = input_probs[:, :, 0][index_map == 0]
    # Necrosis
    heatmap[index_map==50, 0] = input_probs[:, :, 1][index_map==50]
    # Stroma
    heatmap[index_map==100, 0] = input_probs[:, :, 2][index_map==100]
    # ERMS
    heatmap[index_map==150, 0] = input_probs[:, :, 3][index_map==150]
    # ARMS
    heatmap[index_map==200, 0] = input_probs[:, :, 4][index_map==200]
    heatmap[np.average(heatmap, axis=-1)==0, :] = 1.
    return heatmap


# saved as I tried to change the output
def _gray_to_color(input_probs):

    index_map = (np.argmax(input_probs, axis=-1)*50).astype('uint8')
    height = index_map.shape[0]
    width = index_map.shape[1]

    heatmap = np.zeros((height, width, 3), np.float32)

    # Background
    heatmap[index_map == 0, 0] = input_probs[:, :, 0][index_map == 0]
    heatmap[index_map == 0, 1] = input_probs[:, :, 0][index_map == 0]
    heatmap[index_map == 0, 2] = input_probs[:, :, 0][index_map == 0]

    # Necrosis
    heatmap[index_map==50, 0] = input_probs[:, :, 1][index_map==50]
    heatmap[index_map==50, 1] = input_probs[:, :, 1][index_map==50]
    heatmap[index_map==50, 2] = 0.

    # Stroma
    heatmap[index_map==100, 0] = 0.
    heatmap[index_map==100, 1] = input_probs[:, :, 2][index_map==100]
    heatmap[index_map==100, 2] = 0.

    # ERMS
    heatmap[index_map==150, 0] = input_probs[:, :, 3][index_map==150]
    heatmap[index_map==150, 1] = 0.
    heatmap[index_map==150, 2] = 0.

    # ARMS
    heatmap[index_map==200, 0] = 0.
    heatmap[index_map==200, 1] = 0.
    heatmap[index_map==200, 2] = input_probs[:, :, 4][index_map==200]

    heatmap[np.average(heatmap, axis=-1)==0, :] = 1.

    return heatmap


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

        for i in range(heights-1):
            for j in range(widths-1):
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

                # save data to look at
                #if (temp_i>100) and (temp_i<400):
                if (False):
                    np.save('hyun-patch-'+unique_identifier+'-'+str(temp_i)+'_'+str(temp_j)+'.npy', test_patch)
                    print('test_patch shape:', test_patch.shape, 'i:',temp_i,' j:',temp_j)
                    np.save('hyun-tensor-' + unique_identifier + '-' + str(temp_i) + '_' + str(temp_j) + '.npy', test_patch_tensor.cpu())
                    print('test_tensor shape:', test_patch.shape, 'i:', temp_i, ' j:', temp_j)
                    from PIL import Image
                    im = Image.fromarray(test_patch)
                    im.save('hyun-patch-'+str(temp_i)+'_'+str(temp_j)+'.jpeg')

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
        for i in range(heights-1 ):
            for j in range(widths-1):
                prob_map_seg[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE,:] \
                    += np.multiply(linedup_predictions[patch_iter, :, :, :], kernel)
                weight_sum[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE,:] \
                    += kernel
                patch_iter += 1
        #np.save("prob_map_seg.npy",prob_map_seg)
        #np.save('weight_sum.npy',weight_sum)
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
    fileNoExtension = os.path.basename(image_path).split('.')[0]
    dirName = os.path.dirname(image_path)
    numpyFileName = os.path.join(dirName,fileNoExtension+'_prob_map_seg_stack.npy')
    np.save(numpyFileName, prob_map_seg_stack)
    print('prob_map_seg_stack:',prob_map_seg_stack.shape)
    pred_map_final = np.argmax(prob_map_seg_stack, axis=-1)
    print('pred_map_final:',pred_map_final.shape)
    print('pred_map_final dtype:',pred_map_final.dtype)
    pred_map_final_gray = pred_map_final.astype('uint8') * 50
    print('pred_map_final_gray dtype:',pred_map_final_gray.dtype)
    #del pred_map_final
    gc.collect()
    pred_map_final_ones = [(pred_map_final_gray == v) for v in CLASS_VALUES]
    del pred_map_final_gray
    gc.collect()
    pred_map_final_stack = np.stack(pred_map_final_ones, axis=-1).astype('uint8')
    print('pred_map_final_stack shape:',pred_map_final_stack.shape)
    del pred_map_final_ones
    gc.collect()

    pred_labelmap = pred_map_final
    #pred_labelmap = _gray_to_labelmap(pred_map_final_stack)
    #pred_colormap = _gray_to_color(pred_map_final_stack)
    print('pred_labemap:',pred_labelmap.shape)
    del pred_map_final_stack
    gc.collect()

    #np.save('pred_map_final_stack.npy', pred_map_final_stack)
    #prob_colormap = _gray_to_color(prob_map_seg_stack)
    #np.save('prob_colormap.npy', prob_colormap)

    # changing output to not be scaled by 256,  instead each channel is either 0 or 1
    out_label = (pred_labelmap).astype('uint8')
    #out_color = (pred_colormap*255).astype('uint8')

    #numpyFileName = os.path.join(dirName,fileNoExtension+'_out_color.npy')
    #np.save(numpyFileName, out_color)
    # return image instead of saving directly
    return out_label




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


def inference_image(modelWeightFile,model, image_path, BATCH_SIZE, num_classes):
    kernel = _gaussian_2d(num_classes, 0.5, 0.0)
    predict_image = _inference(model, image_path, BATCH_SIZE, num_classes, kernel, 1)
    return predict_image

# -- delete? ----
def start_inference(modelWeightFile,msg_queue, image_file):
    reset_seed(1)

    best_prec1_valid = 0.
    torch.backends.cudnn.benchmark = True

    #saved_weights_list = sorted(glob.glob(WEIGHT_PATH + '*.tar'))
    saved_weights_list = [os.path.join(modelWeightFile,'model_iou_0.4996_0.5897_epoch_45.pth.tar')] 
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

    # return image data so girder toplevel task can write it out
    predict_image = inference_image(model,image_file, BATCH_SIZE, len(CLASS_VALUES))

    # put the filename of the image in the message queue and return it to the main process
    msg_queue.put(predict_image)
    
    # not needed anymore, returning value through message queue
    #return predict_image

def start_inference_mainthread(modelWeightPath,image_file,out_file):
    reset_seed(1)

    best_prec1_valid = 0.
    torch.backends.cudnn.benchmark = True

    #saved_weights_list = sorted(glob.glob(WEIGHT_PATH + '*.tar'))
    saved_weights_list = [os.path.join(modelWeightPath,'model_iou_0.7343_0.7175_epoch_60.pth.tar')] 
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
    predict_image = inference_image(model,image_file, BATCH_SIZE, len(CLASS_VALUES))
    # pass the original dicom file, so header information can be read.  
    # Pass the multichannel segmentation image. 
    writeDicomSegObject(image_file,predict_image,out_file)
    # not needed anymore?
    return predict_image



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


def writeDicomSegObject(image_path, seg_image, out_path):

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path(image_path)

    # Read SM Image data set from PS3.10 files on disk.  This will provide the 
    # reference image size and other dicom header information
    image_dataset = dcmread(str(image_file))

    # function stolen from idc-pan-cancer-archive repository to re-tile the numpy to match the tiling
    # from the source image
    print('passing in a numpy array of shape:',seg_image.shape)
    mask = disassemble_total_pixel_matrix(seg_image,image_dataset)
    print('received a numpy array of shape:',mask.shape)

    
    # Describe the algorithm that created the segmentation
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='FNLCR_RMS_seg_iou_0.7343_epoch_60',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Describe the segment
    description_segment_1 = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=CHANNEL_DESCRIPTION['chan_1'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_1'])
    )
 # Describe the segment
    description_segment_2 = hd.seg.SegmentDescription(
        segment_number=2,
        segment_label=CHANNEL_DESCRIPTION['chan_2'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_2'])
    )

    # Describe the segment
    description_segment_3 = hd.seg.SegmentDescription(
        segment_number=3,
        segment_label=CHANNEL_DESCRIPTION['chan_3'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_3'])
    )

    # Describe the segment
    description_segment_4 = hd.seg.SegmentDescription(
        segment_number=4,
        segment_label=CHANNEL_DESCRIPTION['chan_4'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_4'])
    )

    # Create the Segmentation instance
    seg_dataset = hd.seg.Segmentation(
        source_images=[image_dataset],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        #segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
        segment_descriptions=[description_segment_1,description_segment_2,description_segment_3,description_segment_4],
        series_instance_uid=hd.UID(),
        series_number=2,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        # the following two entries are added because the output resolution is different from the source
        #pixel_measures=derived_pixel_measures,
        #plane_positions= derived_plane_positions,
        manufacturer='Aperio',
        manufacturer_model_name='Unknown',
        software_versions='v1',
        device_serial_number='Unknown'
    )

    #print(seg_dataset)
    # change output file with some function is needed
    outfileanme = out_path
    seg_dataset.save_as(outfileanme)


# a low-res version of one of David's converted outpuots
imagePath = '/media/clisle/Imaging/IDC/pmed_low_res2/DCM_3'
outPath = '/media/clisle/Imaging/IDC/pmed_low_res2'

# a low-res version of one of David's converted outpuots
imagePath = '/media/clisle/KVisImagery/NCI/IDC/Oct2023_RMS_SamplesToIDC/PALMPL-0BMX5D-AKA-RMS2397/image/PALMPL-0BMX5D_1_DCM_3'
outPath = '/media/clisle/KVisImagery/NCI/IDC/Oct2023_RMS_SamplesToIDC/PALMPL-0BMX5D-AKA-RMS2397/model_prediction'
