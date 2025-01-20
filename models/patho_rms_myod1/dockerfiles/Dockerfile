#FROM mhubai/base:latest
FROM mhubbase


# pull the weights for the RMS model from a public repository which is a mirror of the
# original uploaded location (which wasn't available via a direct URL)
ENV WEIGHTS_DIR="/root/.cache/torch/hub/checkpoints/"
ENV MODEL_NAME="myod1_fold_01.pth"
ENV WEIGHTS_URL_PREFIX="https://data.kitware.com/api/v1/file/678684417456515045e2ee60/download"
ENV WEIGHTS_URL=${WEIGHTS_URL_PREFIX}
RUN echo "weights:" ${WEIGHTS_URL}
# download the weights
# *** NOTE: this should be replaced with a call to buildutils/download_weights.sh
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# download EfficientNet pretrained weights
ENV PREWEIGHTS_DIR="/root/.cache/torch/checkpoints/"
RUN wget --directory-prefix ${PREWEIGHTS_DIR} https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth  


# load prerequisites.  There are different versions here compared with the original RMS segmentation model.
RUN echo "loading prerequisite libraries"
RUN pip3 install  torch==1.7.1
RUN pip3 install albumentations
RUN pip3 install segmentation_models_pytorch==0.1.3 --no-dependencies 
RUN pip3 install pretrainedmodels 
RUN pip3 install torchvision==0.8.2 --no-dependencies 
RUN pip3 install efficientnet-pytorch==0.6.3
RUN pip3 install timm==0.3.2
RUN pip install torchnet
RUN pip3 install openslide-python

#RUN pip3 install dicomslide
#RUN pip3 install highdicom

# Import the MHub model definiton
# *** hardcoding for now
ARG MHUB_MODELS_REPO="https://github.com/curtislisle/mhub_models::rms_myod1"
RUN buildutils/import_mhub_model.sh patho_rms_myod1 ${MHUB_MODELS_REPO}


# the weights come bundled in a tar file, so untar
ENV WEIGHTS_ARCHIVE = ${WEIGHTS_DIR}\/${MODEL_NAME}
ENV CURRENT_DIR=`pwd`
WORKDIR ${WEIGHTS_DIR}
RUN mv download ${MODEL_NAME}


# Default run script
#ENTRYPOINT ["python3", "-m", "mhubio.run"]
#CMD ["--config", "/app/models/a3ds-vertebrae/config/default.yml"]
ENTRYPOINT ["/bin/bash"]