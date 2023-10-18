"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Spider baseline Algorithm
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""
from pathlib import Path
import shutil
import os

import SimpleITK
import numpy as np

from mhubio.core import Instance, InstanceData, IO, Module

# The GC Spider baseline algorithm is imported as a self-contained algorithm class with an execute method
from TestSpine import Module as SpiderAlgorithm


class SpiderBaselineRunner(Module):

    SPIDER_DATA_DIR = Path(os.environ["VERSEG_BASEDIR"])
    SPIDER_INTERNAL_DATASET_PATH = SPIDER_DATA_DIR / "datasets" / "spider_input"
    SPIDER_EXPERIMENT_DIR = SPIDER_DATA_DIR / "experiments" / "SPIDER-Baseline"
    SPIDER_INTERNAL_OUTPUT_DIR = SPIDER_EXPERIMENT_DIR / "results" / "spider_input"

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=mr', the='input sagittal spine MRI')
    @IO.Output(
        'out_data_raw',
        'spider_baseline_vertebrae_segmentation_raw.mha',
        'mha:mod=seg:model=SpiderBaselineSegmentation:seg=raw',
        data='in_data',
        bundle='model',
        the='Spider baseline vertebrae segmentation for the input sagittal spine MRI. Raw segmentation output: '
            '0: background '
            '1-25: different vertebrae numbered from the bottom (i.e. L5 = 1) '
            '100: spinal canal '
            '101-125 for partially visible vertebrae '
            '201-225: different intervertebral discs (also numbered from the bottom, i.e. L5/S1 = 201)) '
    )
    @IO.Output(
        'out_data',
        'spider_baseline_vertebrae_segmentation.mha',
        'mha:mod=seg:model=SpiderBaselineSegmentation:seg=remapped:roi=VERTEBRAE_L5,VERTEBRAE_L4,VERTEBRAE_L3,VERTEBRAE_L2,VERTEBRAE_L1,VERTEBRAE_T12,VERTEBRAE_T11,VERTEBRAE_T10,VERTEBRAE_T9,VERTEBRAE_T8,VERTEBRAE_T7,VERTEBRAE_T6,VERTEBRAE_T5,VERTEBRAE_T4,VERTEBRAE_T3,VERTEBRAE_T2,VERTEBRAE_T1,VERTEBRAE_C7,VERTEBRAE_C6,VERTEBRAE_C5,VERTEBRAE_C4,VERTEBRAE_C3,VERTEBRAE_C2,VERTEBRAE_C1',
        data='in_data',
        bundle='model',
        the='Spider baseline vertebrae segmentation for the input sagittal spine MRI. Remapped segmentation output: '
            '0: background '
            '1-25: different vertebrae numbered from the bottom (i.e. L5 = 1) (includes partially visible vertebrae from raw segmentation)'
    )
    def task(self, instance: Instance, in_data: InstanceData, out_data_raw: InstanceData, out_data: InstanceData) -> None:
        in_data_path = Path(in_data.abspath)
        internal_img_path = self.SPIDER_INTERNAL_DATASET_PATH / "images" / "input_img.mha"
        self.log(f"Copying data files to internal SPIDER data structure: {in_data_path} -> {internal_img_path}", level="NOTICE")
        shutil.copy(str(in_data_path), str(internal_img_path))

        self.log("Run the SPIDER-Baseline algorithm", level="NOTICE")
        # The algorithm is configured to run on an internal data folder structure and is
        # further configured by the JSON files in the Dockerfile
        # The algorithm parameters specify the following:
        # * The first argument specifies the folder to look for the model files
        # * The epoch specifies the model weights file to use, which should be something like: `999999.pt`
        # * The dataset is set to the internal SPIDER data structure
        # * The surface erosion threshold is set to the default for the MR modality (as it is always assumed to have MR modality)
        # * The original image dimensions are used to generate the output segmentation
        SpiderAlgorithm(
            [
                "SPIDER-Baseline",
                "--epoch", "999999",
                "--dataset", "spider_input",
                "--surface_erosion_threshold", "-2000",
                "--export_original"
            ]
        ).execute()

        source_output_file = self.SPIDER_INTERNAL_OUTPUT_DIR / "input_img_total_segmentation_original.mha"

        self.log(f"Move raw generated segmentation output from: {source_output_file} -> {out_data_raw.abspath}", level="NOTICE")
        shutil.move(str(source_output_file), out_data_raw.abspath)

        # Create the remamapping dictionary to reorder output labels so they will be picked up in the correct order by DicomSeg
        remap_dict = {i:i for i in range(0, 26)}                    # keep labels 0-25 the same
        remap_dict.update({i:i-100 for i in range(101, 126)})       # partially visible vertebrae get remapped to regular vertebrae labels
        # TODO the following two lines can be added if their rois are added to the segdb
        # remap_dict.update({i:i-201+26 for i in range(201, 226)})  # remaps intervertebral discs to 26-49
        # remap_dict.update({100: 50})                              # remaps spinal canal to 50

        # Convert the mapping to a 1d numpy vector by generating a value for each potential segmentation value
        # Each value is mapped to zero by default and the mapping values are overwritten by the remap_dict
        remap_np = np.zeros((226,), dtype=int)
        remap_np[list(remap_dict.keys())] = list(remap_dict.values())

        self.log(f"Remap generated segmentation output to: {out_data.abspath}", level="NOTICE")
        self.log(f"  mapping used: {remap_dict}", level="DEBUG")
        seg_sitk = SimpleITK.ReadImage(out_data_raw.abspath)
        seg_np = SimpleITK.GetArrayFromImage(seg_sitk)
        seg_remapped_np = remap_np[seg_np]  # actual remapping
        seg_remapped_sitk = SimpleITK.GetImageFromArray(seg_remapped_np)
        seg_remapped_sitk.CopyInformation(seg_sitk)
        SimpleITK.WriteImage(seg_remapped_sitk, out_data.abspath, True)

        # Run internal output cleanup
        shutil.rmtree(str(self.SPIDER_INTERNAL_OUTPUT_DIR))
