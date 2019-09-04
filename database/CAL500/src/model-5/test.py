from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.datasets import dump_svmlight_file
from keras.models import Model, Sequential
from keras.layers import Dense, Activation

import numpy as np
import os


#### 2048-features vector
model = InceptionV3(include_top=False, pooling = 'max', weights='imagenet', input_shape=(256,128,3))

# model.summary()

# img_path = '/gdrive/My Drive/writer_id/Projeto_Fabio/database/iam/segmented-ord/000/a01-000u.png'
# img = image.load_img(img_path, target_size=(299, 299))
# img_data = image.img_to_array(img)
# img_data = np.expand_dims(img_data, axis=0)
# img_data = preprocess_input(img_data)

# feature = model.predict(img_data)

# subdir = '/gdrive/My Drive/writer_id/Projeto_Fabio/database/iam/texture-docQnt/2'
# output = '/gdrive/My Drive/writer_id/Projeto_Fabio/database/feats/incep_iam_forms_2documento_3x3.txt.svm'

# np_data = '/gdrive/My Drive/writer_id/Projeto_Fabio/database/feats/incep_iam_forms_2documento_3x3_data.npy'
# np_label = '/gdrive/My Drive/writer_id/Projeto_Fabio/database/feats/incep_iam_forms_2documento_3x3_label.npy'

# feature_list = []
# target_list  = []
# for classe in os.listdir(subdir):
#     # get the writers ids, i.e., 'CF00000' or 'CF00001' and so on
#     # ...                           would be like '000', '001'...
    
#     for doc in os.listdir(subdir+'/'+classe):
#         # process the docs under the directory 
#         # ...
#         img_path = '{}/{}/{}'.format(subdir,classe,doc)
#         img = image.load_img(img_path, target_size=(256, 128))
#         img_data = image.img_to_array(img)
#         img_data = np.expand_dims(img_data, axis=0)
#         img_data = preprocess_input(img_data)

        
#         feature = model.predict(img_data)
#         feature_np = np.array(feature)
#         feature_list.append(feature_np.flatten())
#         target_list.append(int(classe.split('CF00')[-1]))
        
### zero_based=False ---> vector starts in 1:...                                                    
# dump_svmlight_file(feature_list,target_list, output,zero_based=False,)
# np.save(np_data, feature_list)
# np.save(np_label, target_list) 