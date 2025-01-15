#!/bin/bash

fmnist_root="data/fashion-mnist"
echo ${fmnist_root}


mirror="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
declare -a files=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

for index in "${!files[@]}"
do
    current_file="${fmnist_root}/${files[index]}" 

    # Skip all existed files
    if [ -f ${current_file} ]; then
        echo "${current_file} exists, skipped"
        continue
    fi

    url="${mirror}/${files[index]}.gz"
    wget -P ${fmnist_root} ${url}
    gzip -d ${current_file}
done
