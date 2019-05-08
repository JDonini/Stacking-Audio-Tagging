#!/bin/bash

MARSYAS_RUBY=$(python $HOME/Workspace/Stacking-holdout/utils/config_to_shell.py "MARSYAS_RUBY")
VALIDATION=$(python $HOME/Workspace/Stacking-holdout/utils/config_to_shell.py "VALIDATION")
STACKING=$(python $HOME/Workspace/Stacking-holdout/utils/config_to_shell.py "STACKING")

ruby ${MARSYAS_RUBY}/per-tag-and-global-precision-recall-fixed.rb ${VALIDATION}/${i}/ground_truth.txt ${STACKING}/${i}/stage1_predictions.txt > ${STACKING}/${i}/stage1_evaluation.txt
ruby ${MARSYAS_RUBY}/per-tag-and-global-precision-recall-fixed.rb ${VALIDATION}/${i}/ground_truth.txt ${STACKING}/${i}/stage2_predictions.txt > ${STACKING}/${i}/stage2_evaluation.txt

echo "Computing evaluation ... "

python src/compute_evaluation.py
