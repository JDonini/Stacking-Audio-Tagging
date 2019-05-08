import sys
import os

pwd = os.getcwdb().decode('utf8')
database_name = os.environ["database_name"]

if sys.argv[1] == 'AUDIO':
    print(pwd + '/database/' + database_name + '/data/audio/')

elif sys.argv[1] == 'TRAIN_TXT':
    print(pwd + '/database/' + database_name + '/data/annotations/train.txt')

elif sys.argv[1] == 'TEST_TXT':
    print(pwd + '/database/' + database_name + '/data/annotations/test.txt')

elif sys.argv[1] == 'VALIDATION_TXT':
    print(pwd + '/database/' + database_name + '/data/annotations/ground_truth.txt')

elif sys.argv[1] == 'TRAIN_ARFF':
    print(pwd + '/database/' + database_name + '/out/first_stage/cnn-svm-svm/stacking/train.arff')

elif sys.argv[1] == 'TEST_ARFF':
    print(pwd + '/database/' + database_name + '/out/first_stage/cnn-svm-svm/stacking/test.arff')

elif sys.argv[1] == 'EVALUATION':
    print(pwd + '/database/' + database_name + '/out/first_stage/cnn-svm-svm/evaluation/')

elif sys.argv[1] == 'STACKING':
    print(pwd + '/database/' + database_name + '/out/first_stage/cnn-svm-svm/stacking/')

elif sys.argv[1] == 'MARSYAS_BEXTRACT':
    print('/home/juliano/Workspace/marsyas/build/bin/bextract')

elif sys.argv[1] == 'MARSYAS_RUBY':
    print('/home/juliano/Workspace/marsyas/scripts/Ruby')

elif sys.argv[1] == 'MARSYAS_KEA':
    print('/home/juliano/Workspace/marsyas/build/bin/kea')
sys.exit(0)