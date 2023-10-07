#!/bin/sh

# Install the requirements of TensorFlow Models and this repo:
cd models

# Compile the protos
# If `protoc` is not installed, please follow: https://grpc.io/docs/protoc-installation/
export PYTHONPATH=${PYTHONPATH}:${PWD}/research/
cd research/object_detection/
protoc protos/string_int_label_map.proto --python_out=.

cd ../../..

# Compile the protos
protoc deeplab2/*.proto --python_out=.

# Add to PYTHONPATH the directory where deeplab2 sits.
export PYTHONPATH=${PYTHONPATH}:${PWD}