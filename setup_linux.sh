#!/bin/bash

echo "Welcome to SysId."
echo "Installing system dependencies..."
echo "You will probably be asked for your sudo password."
sudo apt-get update
sudo apt-get install -y python-pip python-dev swig cmake build-essential
sudo apt-get install g++ libblas-dev
sudo apt-get build-dep -y python-scipy
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
pip install --upgrade pip

echo "Creating conda environment..."
conda env create -f environment.yml
conda env update

echo "Conda environment created!"
