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
sed -i -e "s/docker-compose run --rm mykaggle/docker-compose run --rm $1/g" ./script/setup_kaggle_dataset.sh
sed -i -e "s/docker-compose run --rm mykaggle/docker-compose run --rm $1/g" ./script/setup_torch.sh
sed -i -e "s/docker-compose run --rm mykaggle/docker-compose run --rm $1/g" ./script/upload_ckpt.sh
