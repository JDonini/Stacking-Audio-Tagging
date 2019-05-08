#!/usr/bin/env bash

cd $(pwd)/database/CAL500/data
wget http://calab1.ucsd.edu/~datasets/cal500/cal500data/CAL500_32kps.tar
tar -xf CAL500_32kps.tar
rm CAL500_32kps.tar

find CAL500_32kps -name "*.mp3" -exec mv -t raw/audio {} \;
rm -rf CAL500_32kps

cd raw/annotations
wget http://calab1.ucsd.edu/~datasets/cal500/cal500data/cal500_song_tag_annotations.txt
