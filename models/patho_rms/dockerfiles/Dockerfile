#FROM mhubai/base:latest
FROM mhubbase


# pull the weights for the segresnet model trained on the abdominal total
# segmentator data model from the MONAIAuto3DSeg Slicer extension repository
ENV WEIGHTS_DIR="/root/.cache/torch/hub/checkpoints/"
ENV MODEL_NAME="rms_segment_fold_03.pth"
ENV WEIGHTS_URL_PREFIX="https://data.kitware.com/api/v1/file/6675ac9a9876f89773631822/download"
ENV WEIGHTS_URL=${WEIGHTS_URL_PREFIX}
RUN echo "weights:" ${WEIGHTS_URL}
# download the weights
# *** NOTE: this should be replaced with a call to buildutils/download_weights.sh
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}


# load prerequisites 
RUN echo "loading prerequisite libraries"
RUN pip3 install  torch==1.4.0
RUN pip3 install torchvision==0.5.0 
RUN pip3 install efficientnet-pytorch==0.6.3
RUN pip3 install pretrainedmodels==0.7.4
RUN pip3 install albumentations
RUN pip3 install segmentation_models_pytorch==0.1.0 --no-dependencies
RUN pip3 install timm==0.1.18 --no-dependencies

RUN echo 'installing large image'
RUN pip3 install large-image[dicom] --find-links https://girder.github.io/large_image_wheels



RUN pip3 install dicomslide
RUN pip3 install highdicom

# Import the MHub model definiton
# *** hardcoding for now
ARG MHUB_MODELS_REPO="https://github.com/curtislisle/mhub_models::rms"
RUN buildutils/import_mhub_model.sh patho_rms ${MHUB_MODELS_REPO}


# the weights come bundled in a tar file, so untar
ENV WEIGHTS_ARCHIVE = ${WEIGHTS_DIR}\/${MODEL_NAME}
ENV CURRENT_DIR=`pwd`
WORKDIR ${WEIGHTS_DIR}
RUN mv download ${MODEL_NAME}


# Default run script
#ENTRYPOINT ["python3", "-m", "mhubio.run"]
#CMD ["--config", "/app/models/a3ds-vertebrae/config/default.yml"]
ENTRYPOINT ["/bin/bash"]