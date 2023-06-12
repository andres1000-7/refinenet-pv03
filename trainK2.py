# %%capture
import os

from scripts.customGenerator import customGenerator
from scripts.helpers import get_label_info, gen_dirs, save_settings
from scripts.training import step_decay_schedule, ignore_unknown_xentropy, TensorBoardWrapper
from scripts.customCallbacks import LossHistory, OutputObserver
from keras.losses import binary_crossentropy
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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

root_path = "/kaggle/input/solar-panel-detection-and-identification/PV03"
categories_paths = os.listdir(root_path)
categories_paths = [os.path.join(root_path, cat_path) for cat_path in categories_paths]

for cat_path in categories_paths:
    for _, _, files in os.walk(cat_path):
        print("{}: {}".format(cat_path, len(files)))

image_path = '/kaggle/input/solar-panel-detection-and-identification/PV03/PV03_Rooftop/PV03_316370_1203802.bmp'
mask_path = '/kaggle/input/solar-panel-detection-and-identification/PV03/PV03_Rooftop/PV03_316370_1203802_label.bmp'
image = plt.imread(image_path)
mask = np.expand_dims(plt.imread(mask_path), axis=(-1))
image_shape = image.shape
mask_shape = mask.shape

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

axes[0].imshow(image)
axes[0].set_title('Shape: ' + str(image_shape))

axes[1].imshow(mask, cmap="summer")
axes[1].set_title('Shape: ' + str(mask_shape))

[ax.axis("off") for ax in axes]
plt.show()

images_paths = []
for cat_path in categories_paths:
    for root, _, files in os.walk(cat_path):
        cd_images = [os.path.join(root, file) for file in files]
        [images_paths.append(img) for img in cd_images]
images_paths = sorted(images_paths)
print(images_paths[:6])

n_images = len(images_paths)
new_size = (256, 256)
images_idx = range(0, n_images, 2)
train_idx, test_idx = train_test_split(images_idx, test_size=0.15)


def train_dataset_generator():
    for i in train_idx:
        image = (
                tf.convert_to_tensor(plt.imread(images_paths[i]), dtype=tf.float32) / 255.0
        )
        mask = (
                tf.convert_to_tensor(
                    np.expand_dims(plt.imread(images_paths[i + 1]), axis=(-1)),
                    dtype=tf.float32,
                )
                / 255.0
        )

        image = tf.image.resize(image, new_size)
        mask = tf.image.resize(mask, new_size)

        yield image, mask


def test_dataset_generator():
    for i in test_idx:
        image = (
                tf.convert_to_tensor(plt.imread(images_paths[i]), dtype=tf.float32) / 255.0
        )
        mask = (
                tf.convert_to_tensor(
                    np.expand_dims(plt.imread(images_paths[i + 1]), axis=(-1)),
                    dtype=tf.float32,
                )
                / 255.0
        )

        image = tf.image.resize(image, new_size)
        mask = tf.image.resize(mask, new_size)

        yield image, mask

train_dataset = tf.data.Dataset.from_generator(
    train_dataset_generator,
    output_signature=(
        tf.TensorSpec(shape=(*new_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*new_size, 1), dtype=tf.float32),
    ),
)

test_dataset = tf.data.Dataset.from_generator(
    test_dataset_generator,
    output_signature=(
        tf.TensorSpec(shape=(*new_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*new_size, 1), dtype=tf.float32),
    ),
)

def show_images(images, titles=None):
    if not titles:
        titles = [img.shape for img in images]
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(10, 30))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="summer")
        ax.set_title(titles[i])
        ax.axis("off")
    plt.show()

for item in train_dataset.take(1):
    show_images(item)

load_model = False
backbone = 'resnet50'
batch_size = 16

def is_test(x, _):
    return x % 4 == 0

def is_train(x, y):
    return not is_test(x, y)


recover = lambda x, y: y

valid_dataset = train_dataset.enumerate().filter(is_test).map(recover).batch(batch_size)

train_dataset = train_dataset.enumerate().filter(is_train).map(recover).batch(batch_size)

model = build_refinenet(input_shape=(256, 256, 3), num_class=1, resnet_weights=None, frontend_trainable=True)

model.compile("Adam",
              loss=binary_crossentropy,
              metrics=['accuracy'])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            for item in train_dataset.unbatch().shuffle(1).take(1):
                image = item[0]
                mask_4d = self.model.predict(np.expand_dims(image, axis=(0)))
                mask = np.squeeze(mask_4d, axis=0)
                show_images((image, mask))

display_cb = DisplayCallback()
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights='True')




history = model.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=100,
    validation_data=valid_dataset,
    callbacks=[display_cb, early_stopping_cb],
)

history = pd.DataFrame.from_dict(history.history)

history.to_csv("history_unet-mobilnetv2.csv", index=False)
model.save("model_unet-mobilnetv2.h5")

