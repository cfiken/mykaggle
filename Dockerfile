FROM ubuntu:18.04

# 時刻・言語環境設定に必要最小限の apt
RUN apt-get update && apt-get install -y \
    locales \
    tzdata \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Set environment to japanese
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

# Add ASIZ/TOKYO time zone
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list
ENV TZ Asia/Tokyo
RUN echo "${TZ}" > /etc/timezone \
  && rm /etc/localtime \
  && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
  && dpkg-reconfigure -f noninteractive tzdata

# user
ENV USER=ubuntu GROUP=ubuntu
ENV HOME=/home/$USER
ENV PATH=/home/ubuntu/.local/bin:$PATH

# volume マウント時にファイルを扱えるよう、ホストマシンのユーザーと uid, gid を合わせる
RUN groupadd -g 1000 -r $GROUP \
  && useradd --create-home --no-log-init -r -s /bin/zsh -u 1000 -g $GROUP $USER

# apt-packages
RUN apt-get update && apt-get install -y \
    zsh \
    curl \
    git \
    libbz2-dev \
    libcupti-dev \
    libffi-dev \
    liblzma-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    pkg-config \
    zlib1g-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER $USER
WORKDIR $HOME

# ユーザー設定全般
COPY --chown=$USER:$GROUP .zshenv ./
RUN mkdir -p ~/.ssh \
  && ssh-keyscan github.com >> .ssh/known_hosts

# Python
ENV PYTHON_VERSION=3.8.6 \
  POETRY_VIRTUALENVS_IN_PROJECT=1
RUN git clone https://github.com/pyenv/pyenv.git .pyenv \
  && git clone https://github.com/pyenv/pyenv-virtualenv.git .pyenv/plugins/pyenv-virtualenv \
  && . ~/.zshenv \
  && pyenv install $PYTHON_VERSION \
  && pyenv global $PYTHON_VERSION \
  && pyenv rehash \
  && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - --version 1.0.10 \
  && rm -rf $HOME/.cache

USER $USER

WORKDIR /app

CMD python --version
