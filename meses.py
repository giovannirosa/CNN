# from google.colab.patches import cv2_imshow
import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as K
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import pylab as pl
from matplotlib import ticker
# import progressbar

train_file = './train.txt'
test_file = './test.txt'
# input image dimensions
img_rows, img_cols = 64, 64
num_classes = 12


def data_augmentation(path):
    # load the image
    img = load_img(path)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(brightness_range=[0.5, 1.5])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    img_list = []
    for i in range(3):
        # define subplot
        # pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        img_list.append(image)
        # plot raw pixel data
        # pyplot.imshow(image)
    # show the figure
    return img_list


def load_images(image_paths, convert=False, augmentate=False):
    x = []
    y = []
    for image_path in image_paths:
        path, label = image_path.split(' ')
        path = './data/' + path

        if convert:
            image_pil = Image.open(path).convert('RGB')
        else:
            image_pil = Image.open(path).convert('L')

        img = np.array(image_pil, dtype=np.uint8)

        x.append(img)
        y.append([int(label)])

        if augmentate:
            img_list = data_augmentation(path)
            for gen_img in img_list:
                x.append(gen_img)
                y.append([int(label)])

    x = np.array(x)
    y = np.array(y)

    if np.min(y) != 0:
        y = y-1

    return x, y


def load_dataset(train_file, test_file, resize, convert=False, size=(224, 224)):
    arq = open(train_file, 'r')
    texto = arq.read()
    train_paths = texto.split('\n')

    print('Size : ', size)

    train_paths.remove('')  # remove empty lines
    train_paths.sort()

    x_train, y_train = load_images(train_paths, convert, True)

    arq = open(test_file, 'r')
    texto = arq.read()
    test_paths = texto.split('\n')

    test_paths.remove('')  # remove empty lines
    test_paths.sort()
    x_test, y_test = load_images(test_paths, convert)

    if resize:
        print("Resizing images...")
        x_train = resize_data(x_train, size, convert)
        x_test = resize_data(x_test, size, convert)

    if not convert:
        x_train = x_train.reshape(x_train.shape[0], size[0], size[1], 1)
        x_test = x_test.reshape(x_test.shape[0], size[0], size[1], 1)

    print(np.shape(x_train))
    return (x_train, y_train), (x_test, y_test)

# Resize data


def resize_data(data, size, convert):

    if convert:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1], 3))
    else:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1]))

    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(
            size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    print(np.shape(data_upscaled))
    return data_upscaled


def confusion_matrix_save_png(method, cm):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title("Matriz de Confusao - " + method)
    fig.colorbar(cax)
    ax.set_xticklabels(['', '1', '2', '3', '4', '5', '6',
                        '7', '8', '9', '10', '11', '12'])
    ax.set_yticklabels(['', 'janeiro', 'fevereiro', 'marco', 'abril', 'maio', 'junho',
                        'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    pl.xlabel('Predito')
    pl.ylabel('Valor Real')
    pl.savefig(method + '_cm.png', bbox_inches='tight')


def get_mean_size(train_file, test_file):
    arq = open(train_file, 'r')
    texto = arq.read()
    train_paths = texto.split('\n')

    arq = open(test_file, 'r')
    texto = arq.read()
    test_paths = texto.split('\n')

    paths = train_paths + test_paths

    height_mean = 0
    width_mean = 0
    for image_path in paths:
        if image_path == '':
            continue
        path = image_path.split(' ')[0]
        path = './data/' + path
        # print(path)

        img_open = cv2.imread(path)
        # print(img_open.shape)
        height, width, channels = img_open.shape
        if height_mean == 0:
            height_mean = height
        else:
            height_mean = (height_mean + height) / 2
        if width_mean == 0:
            width_mean = width
        else:
            width_mean = (width_mean + width) / 2

    return int(round(height_mean)), int(round(width_mean))


height_mean, width_mean = get_mean_size(train_file, test_file)
print('Height Mean: %d, Width Mean: %d' % (height_mean, width_mean))

# rgb
input_shape = (height_mean, width_mean, 3)
(x_train, y_train), (x_test, y_test) = load_dataset(train_file,
                                                    test_file, resize=True, convert=True, size=(height_mean, width_mean))

# save for the confusion matrix
label = []
for i in range(len(x_test)):
    label.append(y_test[i][0])


print('Normalizing images...')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


batch_size = 128
epochs = 20

# create cnn model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', round(score[0] * 100, 2))
print('Test accuracy:', round(score[1] * 100, 2))

# print model.predict_classes(x_test) #classes predicted
# print model.predict_proba(x_test) #classes probability

pred = []
y_pred = model.predict_classes(x_test)
for i in range(len(x_test)):
    pred.append(y_pred[i])

cm = confusion_matrix(label, pred)
print(cm)
confusion_matrix_save_png('CNN', cm)

###############################################################################
# Show confusions
###############################################################################
# meses = ['janeiro', 'fevereiro', 'marco', 'abril', 'maio', 'junho',
#          'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']


# arq = open(test_file, 'r')
# texto = arq.read()
# test_paths = texto.split('\n')

# test_paths.remove('')  # remove empty lines
# test_paths.sort()

# images = []
# labels = []

# for image_path in test_paths:
#     path, label = image_path.split(' ')
#     path = './data/' + path

#     images.append(path)
#     labels.append(int(label))


# for i in range(len(y_pred)):

#     # Erro...
#     if (y_pred[i] != labels[i]):
#         print(i)

#         print("Label:", meses[labels[i]],
#               " | Prediction:", meses[y_pred[i]], images[i])
#         im = cv2.imread(images[i])
#         cv2.imshow('ImageWindow', im)
#         cv2.waitKey()
