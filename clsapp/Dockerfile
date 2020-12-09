# {{{{{ build phase }}}}}
FROM tensorflow/tensorflow AS builder

ARG USERNAME=serv
# For Linux local machine
#ARG USER_UID=1000
# For MacOS local machine
ARG USER_UID=501
ARG USER_GID=${USER_UID}

ARG APP_DIR=/app
ARG OPT_DIR=/cls/opt

RUN apt-get update && apt-get install -y \
    g++ gcc libc6-dev make pkg-config \
    tree git sudo curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    # [Optional] Add sudo support for the non-root user
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Install TensorFlow C library
RUN curl -L \
    "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz" | \
    tar -C "/usr/local" -xz
RUN ldconfig

RUN [ -d "${APP_DIR}" ] || mkdir -p ${APP_DIR}
RUN [ -d "${OPT_DIR}" ] || mkdir -p ${OPT_DIR}

# golang
RUN curl https://dl.google.com/go/go1.14.4.linux-amd64.tar.gz | tar zx -C ${OPT_DIR}

COPY . ${APP_DIR}

RUN chown -R ${USERNAME} ${APP_DIR}
RUN chown -R ${USERNAME} ${OPT_DIR}

WORKDIR ${APP_DIR}
USER ${USERNAME}

ENV GOPATH ${APP_DIR}/.go
ENV PATH ${APP_DIR}/.go/bin:${OPT_DIR}/go/bin:$PATH

RUN go build -o clsapp

# {{{{{ install phase }}}}}
FROM tensorflow/tensorflow

ARG USERNAME=serv
# For Linux local machine
#ARG USER_UID=1000
# For MacOS local machine
ARG USER_UID=501
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

# Install TensorFlow C library
RUN curl -L \
    "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz" | \
    tar -C "/usr/local" -xz
RUN ldconfig

RUN [ -d "${APP_DIR}" ] || mkdir -p ${APP_DIR}
RUN [ -d "${IMAGEDATA_DIR}" ] || mkdir -p ${IMAGEDATA_DIR}
RUN [ -d "${MODELDATA_DIR}" ] || mkdir -p ${MODELDATA_DIR}

COPY --from=builder ${APP_DIR}/clsapp ${APP_DIR}

RUN chown -R ${USERNAME} ${APP_DIR}
RUN chown -R ${USERNAME} ${IMAGEDATA_DIR}
RUN chown -R ${USERNAME} ${MODELDATA_DIR}

WORKDIR ${APP_DIR}

USER ${USERNAME}

RUN git clone https://github.com/eficode/wait-for.git