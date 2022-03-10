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
    'keras_ocr',
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

all_image_urls = image_utils.load_image_json()
# all_image_urls = [
#     'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
#     'https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg'
# ]

# Get a set of three example images
test_images = []
for url in all_image_urls:
    try:
        test_images.append(keras_ocr.tools.read(url))
    except:
        print("failed to load image: ", url)

#images = [test_images[0], test_images[1]]
images = test_images

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
#prediction_groups = pipeline.recognize(test_images)
prediction_groups = pipeline.recognize(test_images, detection_kwargs={
                                       "batch_size": 1}, recognition_kwargs={"batch_size": 1})

# for i in prediction_groups:
#     for y in i:
#         print(y[0])

# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for i, (ax, image, predictions) in enumerate(zip(axs, images, prediction_groups)):
    keras_ocr.tools.drawAnnotations(
        image=image, predictions=predictions, ax=ax)
    # print(len(predictions))
    # print(type(predictions))
    # print(predictions)
    # print(type(image))
    # print(type(ax))
    print(i, all_image_urls[i])
    for prediction in predictions:
        print(prediction[0])
    # image_utils.save_image(PATH_TO_TEST_IMAGES_OUT_DIR,
    #                        f'{i}.jpg', image)
    image_utils.save_fig(PATH_TO_TEST_IMAGES_OUT_DIR,
                         f'{i}.png', ax)
