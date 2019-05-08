#!/bin/bash

MARSYAS_KEA=$(python $HOME/Workspace/Stacking-holdout/utils/config_to_shell.py "MARSYAS_KEA")
MARSYAS_RUBY=$(python $HOME/Workspace/Stacking-holdout/utils/config_to_shell.py "MARSYAS_RUBY")
STACKING=$(python $HOME/Workspace/Stacking-holdout/utils/config_to_shell.py "STACKING")
TRAIN=$(python $HOME/Workspace/Stacking-holdout/utils/config_to_shell.py "TRAIN")

${MARSYAS_KEA} -m tags -id ${STACKING}/ -od ${STACKING}/ -w train.arff.affinities.arff -tw test.arff.affinities.arff -pr stage2_affinities.txt
${MARSYAS_RUBY}/threshold_binarization.rb ${TRAIN}/train.txt ${STACKING}/stage2_affinities.txt > ${STACKING}/stage2_predictions.txt
