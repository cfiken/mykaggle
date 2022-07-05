#!/bin/bash

if test $CI; then
    docker-compose run mykaggle poetry run pip install -U pip setuptools
    poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
else
    docker-compose run mykaggle poetry run pip install -U pip setuptools
    docker-compose run mykaggle poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
fi

