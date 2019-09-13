import sys
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense
from keras.layers.merge import concatenate
sys.path.append('database')
from config_project import IMG_SIZE


def inception_v3():
    return InceptionV3(include_top=False, pooling='max', weights=None, input_shape=(IMG_SIZE))


def vgg_19():
    return VGG19(include_top=False, pooling='max', weights=None, input_shape=(IMG_SIZE))


def inception_resnet_v2():
    return InceptionResNetV2(include_top=False, pooling='max', weights=None, input_shape=(IMG_SIZE))

