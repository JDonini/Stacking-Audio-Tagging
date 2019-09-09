import sys
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense
from keras.layers.merge import concatenate
sys.path.append('database')
from config_project import IMG_SIZE


def inception_v3_stage_1():
    return InceptionV3(include_top=False, pooling='max', weights='imagenet', input_shape=(IMG_SIZE))


def vgg_19_stage_1():
    return VGG19(include_top=False, pooling='max', weights='imagenet', input_shape=(IMG_SIZE))


def inception_resnet_v2_stage_1():
    return InceptionResNetV2(include_top=False, pooling='max', weights='imagenet', input_shape=(IMG_SIZE))


def inception_v3_stage_2():
    return InceptionV3(include_top=False, pooling='max', input_shape=(IMG_SIZE))


def vgg_19_stage_2():
    return VGG19(include_top=False, pooling='max', input_shape=(IMG_SIZE))


def inception_resnet_v2_stage_2():
    return InceptionResNetV2(include_top=False, pooling='max', input_shape=(IMG_SIZE))


def merge_late_fusion_model_5_stage_1():
    input_arq_1, model_arq_1 = inception_v3_stage_1()
    input_arq_2, model_arq_2 = vgg_19_stage_1()
    input_arq_3, model_arq_3 = inception_resnet_v2_stage_1()

    merge = concatenate([model_arq_1, model_arq_2, model_arq_3])

    hidden_1 = Dense(512, activation='relu')(merge)
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    output = Dense(97, activation='sigmoid')(hidden_2)

    return Model(inputs=[input_arq_1, input_arq_2, input_arq_3], outputs=output)


def merge_late_fusion_model_5_stage_2():
    input_arq_1, model_arq_1 = inception_v3_stage_2()
    input_arq_2, model_arq_2 = vgg_19_stage_2()
    input_arq_3, model_arq_3 = inception_resnet_v2_stage_2()

    merge = concatenate([model_arq_1, model_arq_2, model_arq_3])

    hidden_1 = Dense(512, activation='relu')(merge)
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    output = Dense(97, activation='sigmoid')(hidden_2)

    return Model(inputs=[input_arq_1, input_arq_2, input_arq_3], outputs=output)
