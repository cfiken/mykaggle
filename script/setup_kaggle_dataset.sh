#!/bin/bash
#
# create kaggle datasets to initialize a competition repositoty
# run this in repository root: ./script/setup_kaggle_dataset.sh

if test $CI; then
    poetry run kaggle datasets create -p ./kaggle_datasets_ckpt -r zip
    poetry run kaggle datasets create -p ./kaggle_datasets_git -r zip
else
    docker compose run --rm mykaggle poetry run kaggle datasets create -p ./kaggle_datasets_ckpt -r zip
    docker compose run --rm mykaggle poetry run kaggle datasets create -p ./kaggle_datasets_git -r zip
fi
