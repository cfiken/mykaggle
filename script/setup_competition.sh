#!/bin/bash
#
# setup the repository with given name
# run this in repository root: ./script/setup_competition.sh "competition_name"

if [ $# -ne 1 ]; then
  echo "Competition name is needed as args like:"
  echo "./script/setup_competition.sh \"my_kaggle_competition\""
  exit 1
fi

sed -i -e "s/container_name: mykaggle/container_name: $1/g" ./docker-compose.yml
sed -i -e "s/container_name: mykaggle/container_name: $1/g" ./docker-compose-cpu.yml
sed -i -e "s/git-mykaggle/git-mykaggle-$1/g" ./kaggle_datasets_git/dataset-metadata.json
sed -i -e "s/ckpt-mykaggle/ckpt-mykaggle-$1/g" ./kaggle_datasets_ckpt/dataset-metadata.json
