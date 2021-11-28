FROM tensorflow/tensorflow

ARG USERNAME=serv
ARG USER_UID
ARG USER_GID=${USER_UID}

ARG APP_DIR=/app
ARG IMAGEDATA_DIR=/cls/images
ARG MODELDATA_DIR=/cls/models

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

RUN apt-get update && apt-get install -y \
    g++ gcc libc6-dev make pkg-config \
    tree git sudo curl wget netcat iputils-ping \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    # [Optional] Add sudo support for the non-root user
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN [ -d "${APP_DIR}" ] || mkdir -p ${APP_DIR}
RUN [ -d "${IMAGEDATA_DIR}" ] || mkdir -p ${IMAGEDATA_DIR}
RUN [ -d "${MODELDATA_DIR}" ] || mkdir -p ${MODELDATA_DIR}

COPY . ${APP_DIR}/

RUN chown -R ${USERNAME} ${APP_DIR}
RUN chown -R ${USERNAME} ${IMAGEDATA_DIR}
RUN chown -R ${USERNAME} ${MODELDATA_DIR}

WORKDIR ${APP_DIR}

RUN pip install -r requirements.txt

USER ${USERNAME}

ENTRYPOINT [ "python", "./app.py" ]