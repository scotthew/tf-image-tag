# TODO: try image ocr
# https://www.tensorflow.org/lite/examples/optical_character_recognition/overview
# https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/dr/2
# https://colab.research.google.com/github/tulasiram58827/ocr_tflite/blob/main/colabs/KERAS_OCR_TFLITE.ipynb
# https://github.com/faustomorales/keras-ocr
# https://keras-ocr.readthedocs.io/en/latest/
# Google OCR
# https://cloud.google.com/vision/docs/ocr

import os
import matplotlib.pyplot as plt
import image_utils
import datetime

import keras_ocr

PATH_TO_TEST_IMAGES_OUT_DIR = os.path.join(
    'test_images_out',
    'ocr',
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

all_image_urls = image_utils.load_image_json()

# Get a set of three example images
test_images = []
for url in all_image_urls:
    try:
        test_images.append(keras_ocr.tools.read(url))
    except:
        print("failed to load image: ", url)


# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
images = [test_images[0]]
prediction_groups = pipeline.recognize(images)

# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    print(type(image))
