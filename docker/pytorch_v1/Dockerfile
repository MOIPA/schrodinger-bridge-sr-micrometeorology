FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV HOME=/workspace \
    LD_LIBRARY_PATH=/usr/local/lib/ \
    PIP_NO_CACHE_DIR="off" \
    PIP_DEFAULT_TIMEOUT=100

ARG DEBIAN_FRONTEND="noninteractive"

EXPOSE 6006 8888 5000

WORKDIR $HOME

RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    openssh-client \
    zsh \
    unzip \
    less \
    git \
    wget \
    curl \
    ttf-mscorefonts-installer \
    libeccodes-dev \
    libgeos-dev \
    libgdal-dev \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-dev \
    libx11-dev \
    libgl1-mesa-glx \
    xvfb \
    libxrender1 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN mkdir -p /workspace/tmp \
    && cd /workspace/tmp \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf /workspace/tmp

COPY Pipfile Pipfile.lock ${HOME}/

RUN cd ${HOME} \
    && pip3 freeze > uninstall.txt \
    && pip3 uninstall -y -r uninstall.txt \
    && pip3 install -U pip==24.3.1 pipenv==2024.4.0 \
    && export PATH="$PATH:$HOME/.local/bin" \
    && apt-get purge -y python3-yaml \
    && pipenv sync --system --dev \
    && rm -rf ${HOME}/.cache/pip ${HOME}/.cache/pipenv

COPY .jupyter/ ${HOME}/.jupyter/
COPY entry.sh ${HOME}/entry.sh
COPY known_hosts /root/.ssh/known_hosts
COPY known_hosts $HOME/.ssh/known_hosts

