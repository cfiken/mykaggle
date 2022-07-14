#!/bin/bash
#
# create kaggle datasets to initialize a competition repositoty
# run this in repository root: ./script/setup_kaggle_dataset.sh "competition_name"

if [ $# -ne 1 ]; then
  echo "Competition name is needed as args like:"
  echo "./script/setup_kaggle_dataset.sh \"my_kaggle_competition\""
  exit 1
fi

if test $CI; then
    poetry run kaggle datasets create -p ./kaggle_datasets_ckpt -r zip
    poetry run kaggle datasets create -p ./kaggle_datasets_git -r zip
else
    docker-compose run --rm mykaggle poetry run kaggle datasets create -p ./kaggle_datasets_ckpt -r zip
    docker-compose run --rm mykaggle poetry run kaggle datasets create -p ./kaggle_datasets_git -r zip
fi
