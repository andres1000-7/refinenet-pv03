import os

from scripts.customGenerator import customGenerator
from scripts.helpers import get_label_info, gen_dirs, save_settings
from scripts.training import step_decay_schedule, ignore_unknown_xentropy, TensorBoardWrapper
from scripts.customCallbacks import LossHistory, OutputObserver

from model.refinenet import build_refinenet

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
from zipfile import ZipFile
import keras.backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from PIL import Image


def prepare_dataframe(image_path, name):
    solar_ids = []
    paths = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            paths.append(path)

            solar_id = filename.split(".")[0]
            solar_ids.append(solar_id)

    d = {"id": solar_ids, name: paths}
    df = pd.DataFrame(data=d)
    df = df.set_index('id')
    return df


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# copy all images to image folder and save labels to label folder with same name as correspoding image

root_dir = '/kaggle/input/solar-panel-detection-and-identification/PV03'
resolution = 'PV03'
data_dir = os.path.join(root_dir)  # ,resolution)

image_root = '/kaggle/working/train'
label_root = '/kaggle/working/train_masks'
if not os.path.isdir(image_root):
    os.mkdir(image_root)
if not os.path.isdir(label_root):
    os.mkdir(label_root)

images = list()
labels = list()

for (dirpath, dirnames, filenames) in os.walk(data_dir):
    # img_names += [os.path.join(dirpath, file) for file in filenames]
    images += [os.path.join(dirpath, file) for file in filenames]

labels += [i for i in filter(lambda score: '_label.bmp' in score, images)]
images = [i for i in filter(lambda score: '_label.bmp' not in score, images)]

for img_path in images:
    src_path = img_path
    dst_path = os.path.join(image_root,os.path.basename(img_path))
    img = Image.open(src_path)
    new_img = img.resize( (256, 256) )
    new_img.save( dst_path[:-4]+'.png', 'png')

for label_path in labels:
    src_path = label_path
    file_name = os.path.basename(label_path).replace('_label','')
    dst_path = os.path.join(label_root,file_name)
    img = Image.open(src_path)
    new_img = img.resize( (256, 256) )
    new_img.save( dst_path[:-4]+'.png', 'png')

print("Train set:", len(os.listdir(image_root)))
print("Train masks:", len(os.listdir(label_root)))

df = prepare_dataframe(image_root, "solar_path")
mask_df = prepare_dataframe(label_root, "mask_path")
df["mask_path"] = mask_df["mask_path"]

img_size = [256, 256]


def data_augmentation(solar_img, mask_img):
    if tf.random.uniform(()) > 0.5:
        solar_img = tf.image.flip_left_right(solar_img)
        mask_img = tf.image.flip_left_right(mask_img)

    return solar_img, mask_img


def preprocessing(solar_path, mask_path):
    solar_img = tf.io.read_file(solar_path)
    solar_img = tf.image.decode_jpeg(solar_img, channels=3)
    solar_img = tf.image.resize(solar_img, img_size)
    solar_img = tf.cast(solar_img, tf.float32) / 255.0

    mask_img = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    mask_img = tf.image.resize(mask_img, img_size)
    mask_img = mask_img[:, :, :1]
    mask_img = tf.math.sign(mask_img)

    return solar_img, mask_img


def create_dataset(df, train=False):
    if not train:
        ds = tf.data.Dataset.from_tensor_slices((df["solar_path"].values, df["mask_path"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((df["solar_path"].values, df["mask_path"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)
        ds = ds.map(data_augmentation, tf.data.AUTOTUNE)

    return ds


# Now we will split the dataset into train and test
train_df, valid_df = train_test_split(df, random_state=42, test_size=.25)
train = create_dataset(train_df, train=True)
valid = create_dataset(valid_df)

TRAIN_LENGTH = len(train_df)
BATCH_SIZE = 24
BUFFER_SIZE = 1000

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
valid_dataset = valid.batch(BATCH_SIZE)

# # Let's look the image and it's corresponding mask
# for i in range(5):
#     for image, mask in train.take(i):
#         sample_image, sample_mask = image, mask
#         display([sample_image, sample_mask])


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_loss(in_gt, in_pred):
    return 1 - dice_coef(in_gt, in_pred)


model = build_refinenet(input_shape=(256, 256, 3), num_class=1, resnet_weights=None, frontend_trainable=True)

model.compile(optimizer='adam',
              loss=dice_loss,
              metrics=[dice_coef, 'accuracy'])

for images, masks in train_dataset.take(1):
    for img, mask in zip(images, masks):
        sample_image = img
        sample_mask = mask
        break


def visualize(display_list, save_path=None):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        # SAVE IMAGE
        plt.axis('off')

    if save_path:
        plt.savefig('/kaggle/working' + save_path)  # Save the image to the specified path

    plt.show()


fig_count = 0


def show_predictions(sample_image, sample_mask):
    global fig_count  # Declare fig_count as a global variable
    path = '/output' + str(fig_count)
    fig_count += 1
    pred_mask = model.predict(sample_image[tf.newaxis, ...])
    pred_mask = pred_mask.reshape(img_size[0], img_size[1], 1)
    visualize(display_list=[sample_image, sample_mask, pred_mask], save_path=path)


# show_predictions(sample_image, sample_mask)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)


# Let's observe how the model improves while it is training.
# To accomplish this task, a callback function is defined below.
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            show_predictions(sample_image, sample_mask)


EPOCHS = 30
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=valid_dataset,
                          callbacks=[DisplayCallback(), early_stop])

# save the model
model.save("best_conv.h5")
print("Model saved successfully")

for i in range(5):
    for image, mask in valid.take(i):
        sample_image, sample_mask = image, mask
        show_predictions(sample_image, sample_mask)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(model_history['loss'])
plt.plot(model_history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(model_history['accuracy'])
plt.plot(model_history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# Save the plot as an image
plt.savefig('plot.png')
plt.show()