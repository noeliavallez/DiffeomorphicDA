#!/bin/bash

mkdir datasets

wget https://figshare.com/ndownloader/files/34144674
mv 34144674 datasets/Pollen.zip
wget https://figshare.com/ndownloader/files/34144665
mv 34144665 datasets/Glomeruli.zip
wget https://figshare.com/ndownloader/files/34144662
mv 34144662 datasets/Diatoms.zip

unzip datasets/Diatoms.zip -d datasets/Diatoms
unzip datasets/Glomeruli.zip -d datasets/Glomeruli
unzip datasets/Pollen.zip -d datasets/Pollen

rm datasets/Diatoms.zip datasets/Glomeruli.zip datasets/Pollen.zip

