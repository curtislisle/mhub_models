FROM mhubai/base:latest

# load prerequisites 
RUN echo "loading prerequisite libraries"
RUN pip3 install  "monai-weekly[nibabel,tqdm]"

RUN echo 'model defined remotely again .....'

# Import the MHub model definiton
# *** hardcoding for now
ARG MHUB_MODELS_REPO="https://github.com/curtislisle/mhub_models"
RUN buildutils/import_mhub_model.sh a3ds-vertebrae ${MHUB_MODELS_REPO}

# pull the weights for the segresnet model trained on the abdominal total
# segmentator data model from the MONAIAuto3DSeg Slicer extension repository
ENV WEIGHTS_DIR="/root/.cache/torch/hub/checkpoints/"
ENV MODEL_NAME="vertebrae-3mm-v2.0.1.zip"
ENV WEIGHTS_URL_PREFIX="https://github.com/lassoan/SlicerMONAIAuto3DSeg/releases/download/Models/"
ENV WEIGHTS_URL=${WEIGHTS_URL_PREFIX}\/${MODEL_NAME}
RUN echo "weights:" ${WEIGHTS_URL}

# download the weights
# *** NOTE: this should be replaced with a call to buildutils/download_weights.sh
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
# the weights come bundled in a zip file, so unzip
ENV WEIGHTS_ARCHIVE = ${WEIGHTS_DIR}\/${MODEL_NAME}
ENV CURRENT_DIR=`pwd`
WORKDIR ${WEIGHTS_DIR}
RUN unzip ${MODEL_NAME}


# Default run script
#ENTRYPOINT ["python3", "-m", "mhubio.run"]
#CMD ["--config", "/app/models/a3ds-vertebrae/config/default.yml"]
ENTRYPOINT ["/bin/bash"]