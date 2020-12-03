FROM tensorflow/tensorflow

ARG USERNAME=dev
# For Linux local machine
#ARG USER_UID=1000
# For MacOS local machine
ARG USER_UID=501
ARG USER_GID=${USER_UID}

ARG OPT_DIR=/cls/opt
ARG IMAGEDATA_DIR=/cls/images
ARG MODELDATA_DIR=/cls/models

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

RUN apt-get update && apt-get install -y \
    g++ gcc libc6-dev make pkg-config \
    tree git sudo curl wget \
    python3-venv python3-matplotlib \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    # [Optional] Add sudo support for the non-root user
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # Avoiding extension reinstalls on container rebuild
    && mkdir -p /home/${USERNAME}/.vscode-server \
    /home/${USERNAME}/.vscode-server-insiders \
    && chown -R ${USERNAME} \
    /home/${USERNAME}/.vscode-server \
    /home/${USERNAME}/.vscode-server-insiders

RUN [ -d "${OPT_DIR}" ] || mkdir -p ${OPT_DIR}
RUN [ -d "${IMAGEDATA_DIR}" ] || mkdir -p ${IMAGEDATA_DIR}
RUN [ -d "${MODELDATA_DIR}" ] || mkdir -p ${MODELDATA_DIR}

COPY requirements.txt ${OPT_DIR}/

RUN chown -R ${USERNAME} ${OPT_DIR}
RUN chown -R ${USERNAME} ${IMAGEDATA_DIR}
RUN chown -R ${USERNAME} ${MODELDATA_DIR}

RUN pip install -r ${OPT_DIR}/requirements.txt