# Define arguments for flexibility
ARG CUDA_VERSION=12.3.1
ARG FLAVOR=devel
ARG OS_VERSION=ubuntu22.04
ARG TAG=${CUDA_VERSION}-${FLAVOR}-${OS_VERSION}


# Base image
FROM nvidia/cuda:${TAG}

ARG PYTHON_VERSION=3.11
WORKDIR /app

# Install Python and update pip
RUN apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python3-pip && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    python -m pip install --upgrade pip

# Install additional packages
RUN apt-get install -y git poppler-utils

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

ENV PYTHONPATH /app
COPY . . 

CMD [ "bash" ]