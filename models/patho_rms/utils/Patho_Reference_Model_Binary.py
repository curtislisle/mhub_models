"""
-------------------------------------------------
MHub - run the NCI RMS segmentation pipeline
-------------------------------------------------

-------------------------------------------------
---------------------------------------------------
Author: Curtis Lisle
Email:  clisle@knowledgevis.com
based on examples from Dennis Bontempi and Leonard Nuernburg.

This is a direct port of the original reference RMS segmentation implementation
developed and trained by Dr. Hyun Jung, Fred. Nat. Lab. for Cancer Research.

This version of the module outputs a binary DICOM segmentation image. The output
contains four channels (ARMS, ERMS, STROMA, NECROSIS).

github location for the original reference code from FNLCR:
https://github.com/abcsFrederick/RMS_AI

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

import random
import argparse
import torch
import torch.nn as nn
import os, glob
import segmentation_models_pytorch as smp
import yaml
import numpy as np
from skimage.io import imread, imsave
from skimage import filters
from skimage.color import rgb2gray
import albumentations as albu
import gc


from PIL import Image
Image.MAX_IMAGE_PIXELS = None


#------ Start Global definitiona ---------------------

BATCH_SIZE = 80

# define global variable that is set according to whether GPUs are discovered
if torch.cuda.is_available():
    USE_GPU = True
    print('GPU is available')
else:
    USE_GPU = False
    print('GPU is not available. Using CPU')


# these aren't used in the girder version, no files are directly written out 
# by the routines written by FNLCR (Hyun Jung)
WSI_PATH = '.'
PREDICTION_PATH = '.'



#------ End Global definitiona ---------------------


class Patho_Reference_RMS_Runner_Binary(ModelRunner):
    #fractional : bool
    #batchsize: int

    @IO.Instance()
    @IO.Input('image', 'png:mod=sm',  the='input whole slide image')
    @IO.Output('structures', 'pathology_rms.seg.png', 'png:mod=seg:model=patho_rms', bundle='model', the='predicted tissue classes')
    def task(self, instance: Instance, image: InstanceData, structures: InstanceData) -> None:
        self.log.debug("Running the segmentation.")
        print('binary segmentation mode is enabled')

        # *** hardcode the model weights location until figuring out
        # the initialization method
        #self.log.debug('Need to pull weights from public repository!')
        #modelCheckpointFilePath = '/home/clisle/proj/slicer/PW39/rms-infer-code-standalone/'
        modelCheckpointFilePath = '/root/.cache/torch/hub/checkpoints/'

        # the input image passed is the containing directory, not a file, so look up a file
        inputImagePath = self.findWsiFile(image.abspath)
        self.log.debug(f'discovered input file is {inputImagePath}')
        self.log.debug(f'output path is {structures.abspath}')
       
        reset_seed(1)
        num_classes = 5
        input_path = inputImagePath
        output_path = structures.abspath
        image_size = 340

        best_prec1_valid = 0.

        if USE_GPU:
            torch.backends.cudnn.benchmark = True

        batch_size = BATCH_SIZE

        ## Three different networks ensembled. You can use other network using command line "--k 1 or 2 or 3"
        #weight_path = parsed_yaml['segment']['weight'] + 'Fold_' + '%02d' % (args.kfold) + '/'
        #saved_weights_list = sorted(glob.glob(weight_path + '*.tar'))
        saved_weights_list = [modelCheckpointFilePath]
        print(saved_weights_list)

        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None,
            aux_params=None,
        )

        if USE_GPU:
            model = nn.DataParallel(model)
            model = model.cuda()
        model = load_best_model(model, saved_weights_list[-1], best_prec1_valid)
        print('Loading model is finished!!!!!!!')

        inference_WSIs(model, image_size, batch_size, input_path, output_path, num_classes)



    # look in the directory for a whole slide image file
    def findWsiFile(self, dirPath):
        file_list = []
        for root, dirs, files in os.walk(dirPath):
            for file in files:
                file_list.append(os.path.join(root, file))
        # *** this is a hack, need to find the right file
        return file_list[0]


# supporting subroutines
#-----------------------------------------------------------------------------


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


#---------------- main inferencing routine ------------------



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

def _generate_th(image_org):
    org_height = image_org.shape[0]
    org_width = image_org.shape[1]

    otsu_seg = np.zeros((org_height//4, org_width//4), np.uint8)

    aug = albu.Resize(p=1.0, height=org_height // 4, width=org_width // 4)
    augmented = aug(image=image_org)
    thumbnail = augmented['image']

    thumbnail_gray = rgb2gray(thumbnail)
    val = filters.threshold_otsu(thumbnail_gray)
    otsu_seg[thumbnail_gray <= val] = 255

    aug = albu.Resize(p=1.0, height=org_height, width=org_width)
    augmented = aug(image=otsu_seg, mask=otsu_seg)
    otsu_seg = augmented['mask']

    print('Otsu segmentation finished')

    return otsu_seg

def _infer_batch(model, test_patch, num_classes):
    ml = nn.Softmax(dim=1)
    with torch.no_grad():
        logits_all = model(test_patch[:, :, :, :])
        logits = logits_all[:, 0:num_classes, :, :]
    prob_classes_int = ml(logits)
    prob_classes_all = prob_classes_int.cpu().numpy().transpose(0, 2, 3, 1)

    return prob_classes_all

def _inference(model, IMAGE_SIZE, BATCH_SIZE, WSI_PATH, PREDICTION_PATH, num_classes, kernel):
    CLASS_VALUES = [0, 50, 100, 150, 200]

    model.eval()

    wsi_list = sorted(glob.glob(WSI_PATH + '*.png'))

    print('Total number of inferencing images: ', len(wsi_list))

    test_list = sorted(wsi_list, key=os.path.getsize)

    for a in range(len(test_list)):
        image_working = imread(test_list[a])
        height_org = image_working.shape[0]
        width_org = image_working.shape[1]

        basename_string = os.path.splitext(os.path.basename(test_list[a]))[0]
        print('Basename String: ', basename_string)

        otsu_org = _generate_th(image_working)//255

        height = image_working.shape[0]
        width = image_working.shape[1]

        PATCH_OFFSET = IMAGE_SIZE * 8
        SLIDE_OFFSET = IMAGE_SIZE // 4

        heights = (height+ PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET + 1
        widths = (width+ PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET + 1

        height_ext = SLIDE_OFFSET * heights + PATCH_OFFSET * 2
        width_ext = SLIDE_OFFSET * widths + PATCH_OFFSET * 2

        org_slide_ext = np.ones((height_ext, width_ext, 3), np.uint8) * 255
        otsu_ext = np.zeros((height_ext, width_ext), np.uint8)
        prob_map_seg = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)
        weight_sum = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)

        org_slide_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, 0:3] = image_working[:, :, 0:3]
        otsu_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width] = otsu_org[:, :]

        linedup_predictions = np.zeros((heights*widths, IMAGE_SIZE, IMAGE_SIZE, num_classes), dtype=np.float32)
        linedup_predictions[:, :, :, 0] = 1.0
        test_patch_tensor = torch.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=torch.float).cuda(non_blocking=True)

        patch_iter = 0
        inference_index = []
        position = 0
        for i in range(heights):
            for j in range(widths):
                test_patch = org_slide_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, 0:3]
                otsu_patch = otsu_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE]
                if np.sum(otsu_patch) > int(0.05*IMAGE_SIZE*IMAGE_SIZE):
                    inference_index.append(patch_iter)
                    test_patch_tensor[position, :, :, :] = torch.from_numpy(test_patch.transpose(2, 0, 1)
                                                                     .astype('float32')/255.0)
                    position += 1
                patch_iter+=1

                if position==BATCH_SIZE:
                    batch_predictions = _infer_batch(model, test_patch_tensor, num_classes)
                    for k in range(BATCH_SIZE):
                        linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

                    position = 0
                    inference_index = []

        # Very last part of the region
        batch_predictions = _infer_batch(model, test_patch_tensor, num_classes)
        for k in range(position):
            linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

        patch_iter = 0
        for i in range(heights):
            for j in range(widths):
                prob_map_seg[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, :] \
                                += np.multiply(linedup_predictions[patch_iter, :, :, :], kernel)
                weight_sum[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE, j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, :] \
                                += kernel
                patch_iter += 1

        prob_map_seg = np.true_divide(prob_map_seg, weight_sum)
        prob_map_valid = prob_map_seg[PATCH_OFFSET:PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, :]

        pred_map_final = np.argmax(prob_map_valid, axis=-1)
        pred_map_final_gray = pred_map_final.astype('uint8') * 50
        pred_map_final_ones = [(pred_map_final_gray == v) for v in CLASS_VALUES]
        pred_map_final_stack = np.stack(pred_map_final_ones, axis=-1).astype('uint8')

        prob_colormap = _gray_to_color(prob_map_valid)
        imsave(PREDICTION_PATH + basename_string + '_prob.png', (prob_colormap * 255.0).astype('uint8'))

        pred_colormap = _gray_to_color(pred_map_final_stack)
        imsave(PREDICTION_PATH + basename_string + '_pred.png', (pred_colormap*255.0).astype('uint8'))
        gc.collect()

def _gaussian_2d(num_classes, image_size, sigma, mu):
    x, y = np.meshgrid(np.linspace(-1, 1, image_size), np.linspace(-1, 1, image_size))
    d = np.sqrt(x * x + y * y)
    # sigma, mu = 1.0, 0.0
    k = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    k_min = np.amin(k)
    k_max = np.amax(k)

    k_normalized = (k - k_min) / (k_max - k_min)
    k_normalized[k_normalized<=1e-6] = 1e-6

    kernels = [(k_normalized) for i in range(num_classes)]
    kernel = np.stack(kernels, axis=-1)

    print('Kernel shape: ', kernel.shape)
    print('Kernel Min value: ', np.amin(kernel))
    print('Kernel Max value: ', np.amax(kernel))

    return kernel

def inference_WSIs(model, image_size, batch_size, input_path, output_path, num_classes):
    kernel = _gaussian_2d(num_classes, image_size, 0.5, 0.0)
    _inference(model, image_size, batch_size, input_path, output_path, num_classes, kernel)

