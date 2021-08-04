#!/bin/bash 
set -e


echo "Downloading Emotion6 dataset"
URL="http://chenlab.ece.cornell.edu/people/kuanchuan/publications/Emotion6.zip"
ZIP_FILE="./datasets/Emotion6.zip"
TARGET_DIR="./datasets/Emotion6/"

mkdir -p "./datasets"
wget -N "$URL" -O "$ZIP_FILE"
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm "$ZIP_FILE"
