#!/bin/bash
#
# Upload ckpt files to kaggle datasets
# run this in repository root: ./script/upload_data.sh "version message"

if [ $# -ne 1 ]; then
  echo "Message is needed as args like:"
  echo "./script/upload_ckpt.sh \"version message\""
  exit 1
fi

if test $CI; then
    poetry run kaggle datasets version -p ./kaggle_datasets_ckpt -r zip -m $1
else
    docker-compose run --rm mykaggle poetry run kaggle datasets version -p ./kaggle_datasets_ckpt -r zip -m $1
fi
