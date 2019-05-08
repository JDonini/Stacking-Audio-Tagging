#!/usr/bin/env bash

python src/convert_to_arff.py -i database/CAL500/out/first_stage/cnn-svm-svm/predictions_test.csv -o database/CAL500/out/first_stage/cnn-svm-svm/stacking/test.arff
python src/convert_to_arff.py -i database/CAL500/out/first_stage/cnn-svm-svm/predictions_train.csv -o database/CAL500/out/first_stage/cnn-svm-svm/stacking/train.arff
