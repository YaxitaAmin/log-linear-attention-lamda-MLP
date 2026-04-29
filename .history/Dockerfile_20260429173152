# syntax = docker/dockerfile:1.0-experimental
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Define build argument for repository name
ARG REPO_BASE=log-linear-attention

# working directory
WORKDIR /workspace

# ---------------------------------------------
# Project-agnostic System Dependencies
# ---------------------------------------------
RUN \
    # Install System Dependencies
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        unzip \
        psmisc \
        vim \
        git \
        ssh \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------
# Build Python depencies and utilize caching
# ---------------------------------------------
COPY ./requirements.txt /workspace/main/${REPO_BASE}/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/main/${REPO_BASE}/requirements.txt

# upload everything
COPY . /workspace/main/${REPO_BASE}/

# Set HOME
ENV HOME="/workspace/main"

# Reset Entrypoint from Parent Images
# https://stackoverflow.com/questions/40122152/how-to-remove-entrypoint-from-parent-image-on-dockerfile/40122750
ENTRYPOINT []

# load bash
CMD /bin/bash