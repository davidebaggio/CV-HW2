#!/bin/bash

if [ ! -d "libtorch" ]; then
    wget https://download.pytorch.org/libtorch/test/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
    unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
    rm libtorch-cxx11-abi-shared-with-deps-latest.zip
fi

cmake -B build
make -j6 -C build
