#!/bin/bash

MARSYAS_KEA=$(python $HOME/Workspace/stacking/database/config_shell.py "MARSYAS_KEA")
MARSYAS_RUBY=$(python $HOME/Workspace/stacking/database/config_shell.py "MARSYAS_RUBY")
STACKING=$(python $HOME/Workspace/stacking/database/config_shell.py "STACKING")

${MARSYAS_KEA} -m tags -id ${STACKING} -od ${STACKING} -w train.arff -tw test.arff -pr stage1_affinities.txt