#!/bin/bash

# Download libtorch if not already present
if [ ! -d "libtorch" ]; then
    wget https://download.pytorch.org/libtorch/test/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
    unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
    rm libtorch-cxx11-abi-shared-with-deps-latest.zip
fi

# Download nlohmann/json.hpp into include/ if not already present
if [ ! -f "include/json.hpp" ]; then
    wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -O include/json.hpp
fi

# Download our trained model if not already present
if [ ! -f "simple_card_classifier_traced.pt" ]; then
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1jSCMt6EG0g56-9XafxIkm-zRKy2Y-9aP" -O simple_card_classifier_traced.pt
fi

# Build the project
cmake -B build
make -j6 -C build
