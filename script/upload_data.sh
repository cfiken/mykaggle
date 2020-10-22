#!/bin/sh
#
# Upload ckpt files to kaggle datasets
# run this in repository root: ./script/upload_data.sh "version message"

if [ $# -ne 1 ]; then
  echo "message is needed as args"
  exit 1
fi

poetry run kaggle datasets version -p ./kaggle_datasets_ckpt -r zip -m $1
