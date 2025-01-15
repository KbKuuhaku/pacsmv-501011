#!/bin/bash

mnist_root="data/mnist"
echo ${fmnist_root}


mirror="https://storage.googleapis.com/cvdf-datasets/mnist"
declare -a files=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

for index in "${!files[@]}"
do
    current_file="${mnist_root}/${files[index]}" 

    # Skip all existed files
    if [ -f ${current_file} ]; then
        echo "${current_file} exists, skipped"
        continue
    fi

    url="${mirror}/${files[index]}.gz"
    wget -P ${mnist_root} ${url}
    gzip -d ${current_file}
done
