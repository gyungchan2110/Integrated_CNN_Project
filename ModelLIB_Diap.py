from keras import backend as K

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D, MaxPooling2D,Dropout,Flatten, ZeroPadding2D, Convolution2D,MaxPooling2D, UpSampling2D
from keras.layers import Input, AveragePooling2D, Dropout, Flatten, merge, Reshape, Activation, GlobalMaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.layers.merge import concatenate
from custom_layers.scale_layer import Scale
from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet152 import ResNet152
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception

from keras.optimizers import SGD
from keras.optimizers import Adam

from se_inception_resnet_v2 import SEInceptionResNetV2
import Config



def Make_Model(modelConfig, datasetConfig):
    
    strModelType = modelConfig.MODEL_TYPE
    strPretrained = modelConfig.PRETRAINED_MODEL
    im_Shape = datasetConfig.IMG_SHAPE
    strOptimizer = modelConfig.OPTIMIZER
    num_Classes = datasetConfig.NUM_CLASS
    learingRate = modelConfig.LEARNING_RATE
    decay = modelConfig.DECAY
    momentum = modelConfig.MOMENTUM
    loss = modelConfig.LOSS

    optimizer = None 

    if(strOptimizer == "SGD"):
        optimizer = SGD(lr=learingRate, decay=decay, momentum=momentum, nesterov=True) # decay = 1e-4
    elif(strOptimizer == "ADAM"):
        optimizer = Adam(lr=learingRate, decay=decay)
    else:
        print("No Such a Optimizer") 
        return None

    model = None 

    Num = 2

    if(strModelType == "VGG16"):
        model = VGG16(weights=strPretrained, include_top=False, input_shape=im_Shape,classes = Num)
    elif(strModelType == "RESNET50"):
        model = ResNet50(weights=strPretrained, include_top=True, input_shape=im_Shape,classes = Num)
    elif(strModelType == "RESNET152"):
        model = build_Resnet152_Model(im_Shape, Num, strPretrained)
    elif(strModelType == "INCEPTIONV3"):
        model = InceptionV3(weights=strPretrained, include_top=True, input_shape=im_Shape,classes = Num)
    elif(strModelType == "INCEPTIONRESV2"):
        model = InceptionResNetV2(weights=strPretrained, include_top=True, input_shape=im_Shape,classes = Num)
    elif(strModelType == "SEINCEPTIONRESV2"):
        model = SEInceptionResNetV2(weights=strPretrained, include_top=True, input_shape=im_Shape,classes = Num)
    elif(strModelType == "XCEPTION"):
        model = Xception(weights=strPretrained, include_top=True, input_shape=im_Shape,classes = Num)
        #basemodel = Xception(weights="imagenet", include_top=True, input_shape=(299,299,3), classes = 1000)
        #x = Dense(num_Classes, activation='softmax', name='predictions')(basemodel.layers[-2].output)
        #model = Model(basemodel.input, x)
    elif(strModelType == "UNET2D"):
        model = build_UNet2D_4L(im_Shape, strPretrained)
    elif(strModelType == "CNN6Layers"):    
        model = build_CNN_6layers(im_Shape, num_classes = num_Classes)
    else:
        print("No Such Model Type") 
        return None

    # for layer in model.layers[:-15] : 
    #     layer.trainable = False

    # oldlayer = model.layers.pop(-1)
    # model.layers.pop(-1)
    x_xc = model.get_layer(index = -2).output
    #print(x_xc.output_shape)
    # print(x_xc.shape)
    #x_xc = GlobalMaxPooling2D()(x_xc)
    out = Dense(int(num_Classes), activation='softmax', name='pp')(x_xc)
   
    model = Model (input = model.input, output = out)

    # upperlayer = model.layers[-2]
    # output = Dense(int(num_Classes), activation='softmax')(upperlayer)
    #model.trainable = False
    #model = model (input = model.layers[0], output = output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    return model



def build_Resnet152_Model(im_shape, num_Classes, pretrained):
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5
     
    #weights_path = '/home/user/zyong/darknetnew/yong/yong/pretrained/resnet152/resnet152_weights_tf.h5'
    # Handle Dimension Ordering for different backends
    global bn_axis

    bn_axis = 3
    img_input = Input(shape=im_shape, name='input')
   
             
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
 
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
 
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))
 
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))
 
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
 
    x_fc = AveragePooling2D((int(im_shape[0]/32), int(im_shape[1]/32)), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(int(num_Classes), activation='softmax', name='fc1000')(x_fc)
 
    model = Model(img_input, x_fc)
    
    return model

def build_UNet2D_4L(im_shape, pretrained):
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    k_size = 3
    data = Input(shape=im_shape)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up1)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up2)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up3)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up4)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged4)

    conv10 = Convolution2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)

    return model

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def Jaccard_coef(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1', bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def build_CNN_6layers(im_shape, num_classes = 2, k_size=3):
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=im_shape)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    faltten = Flatten()(pool3)
    x_fc = Dense(int(num_classes), activation='softmax', name='fc1000')(faltten)
    print(type(x_fc))
    model = Model(data, x_fc)

    return model