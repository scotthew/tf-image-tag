# https://www.tensorflow.org/hub/tutorials/object_detection
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#python-package-installation
# For running inference on the TF-Hub module.
import datetime
import io
import os
from pickletools import uint8
import tensorflow as tf

import tensorflow_hub as hub
from google.protobuf import text_format
from protos import string_int_label_map_pb2

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

import json

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" %
      tf.test.gpu_device_name())

image_urls = [
    # Source: https://commons.wikimedia.org/wiki/File:Baegle_dwa.jpg
    "https://upload.wikimedia.org/wikipedia/commons/f/fc/Baegle_dwa.jpg",
    # By "Michael Miley, Source: https://www.flickr.com/photos/mike_miley/4678754542/in/photolist-88rQHL-88oBVp-88oC2B-88rS6J-88rSqm-88oBLv-88oBC4
    "https://live.staticflickr.com/4009/4678754542_fd42c6bbb8_b.jpg",
    # By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
    "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg",
    # Source: https://commons.wikimedia.org/wiki/File:The_Coleoptera_of_the_British_islands_(Plate_125)_(8592917784).jpg
    "https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg",
    # By Américo Toledano, Source: https://commons.wikimedia.org/wiki/File:Biblioteca_Maim%C3%B3nides,_Campus_Universitario_de_Rabanales_007.jpg
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg",
    # Source: https://commons.wikimedia.org/wiki/File:The_smaller_British_birds_(8053836633).jpg
    "https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg",
]


# https://tfhub.dev/tensorflow/collections/object_detection/1
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
all_modules = {
    # FasterRCNN+InceptionResNet V2: high accuracy,
    "inception_resnet_v2": {
        "module_handle": "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
        "signatures": 'default',
        "module_values": {
            "class_key": "detection_class_entities",
            "score_key": "detection_scores",
            "dtype": tf.float32,
            "min_score": 0.1,
            "max_boxes": 20,
            "result_type": 1,
            "path_to_labels": None,
        },
    },
    # ssd+mobilenet V2: small and fast.
    "mobilenet_v2": {
        "module_handle": "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1",
        "signatures": 'default',
        "module_values": {
            "class_key": "detection_class_entities",
            "score_key": "detection_scores",
            "dtype": tf.float32,
            "min_score": 0.1,
            "max_boxes": 20,
            "result_type": 1,
            "path_to_labels": None,
        },
    },
    # SSD with EfficientNet-b7 + BiFPN feature extractor, shared box predictor and focal loss
    "efficientdet_d7": {
        "module_handle": "https://tfhub.dev/tensorflow/efficientdet/d7/1",
        "signatures": None,
        "module_values": {
            "class_key": "detection_classes",
            "score_key": "detection_scores",
            "dtype": tf.uint8,
            "min_score": 0.1,
            "max_boxes": 20,
            "result_type": 2,
            "path_to_labels": os.path.join('data', 'mscoco_label_map.pbtxt'),
        },
    },
}

def load_image_json():
  image_file_path = os.getenv('IMAGE_FILE_PATH')
  if image_file_path is not None:
    print("loading image from file path: ", image_file_path)
    with open(image_file_path, 'r') as fh:
      image_urls_json = json.load(fh)
    
    return image_urls + image_urls_json['images']
  else:
    return image_urls

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)
  return fig


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1, category_index=None):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/Roboto-Regular.ttf",
                              size=32)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    score = scores[i]
    box = boxes[i]
    if category_index is not None:
      class_name = category_index[class_names[i]]['name']
      class_string = str(class_name)
    else:
      class_name = class_names[i]
      class_string = class_name.decode("ascii")
      
    if score >= min_score:
      ymin, xmin, ymax, xmax = tuple(box)
      display_str = "{}: {}%".format(class_string,
                                     int(100 * score))
      color = colors[hash(class_name) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img


def save_image(image_path, image_np):
    isExist = os.path.exists(PATH_TO_TEST_IMAGES_OUT_DIR)
    if not isExist:
      os.makedirs(PATH_TO_TEST_IMAGES_OUT_DIR)

    # Convert to image and write to out file
    out_image_path = os.path.join(
        PATH_TO_TEST_IMAGES_OUT_DIR, os.path.basename(image_path))
    print("Out Image Name: %s" % (out_image_path))
    out_img = Image.fromarray(image_np)
    out_img.save(out_image_path, "jpeg")
    return


def run_detector(detector, path, module_values, category_index=None):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, module_values["dtype"])[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  if module_values["result_type"] == 1:
    normalized_boxes = result["detection_boxes"]
    normalized_scores = result[module_values["score_key"]]
    normalized_class_names = result[module_values["class_key"]]
    normalized_detection_scores = result["detection_scores"]
  else:
    normalized_boxes = result["detection_boxes"][0]
    normalized_scores = result[module_values["score_key"]][0]
    normalized_class_names = result[module_values["class_key"]][0]
    normalized_detection_scores = result["detection_scores"][0]


  #print("detection scores: ", result["detection_scores"])
  print("Found %d objects." % len(normalized_detection_scores))
  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
    image=img.numpy(),
    boxes=normalized_boxes,
    class_names=normalized_class_names,
    scores=normalized_scores,
    max_boxes=module_values["max_boxes"],
    min_score=module_values["min_score"],
    category_index=category_index,
  )
  save_image(path, image_with_boxes)

  # TODO: Combine images into summary
  #figure = display_image(image_with_boxes)
  #filename = os.path.basename("path/to/file/sample.txt")
  #with file_writer.as_default():
  #  tf.summary.image(filename, plot_to_image(figure), step=0)


def detect_img(image_path, module_values, category_index=None):
  start_time = time.time()
  run_detector(detector, image_path, module_values, category_index)
  end_time = time.time()
  print("Inference time:", end_time-start_time)


def load_category_index(path_to_labels=None):
  if path_to_labels is not None:
    with open(path_to_labels, 'r') as f:
      label_map_string = f.read()
      label_map = string_int_label_map_pb2.StringIntLabelMap()
      text_format.Merge(label_map_string, label_map)

    category_index = {}

    for item in label_map.item:
      name = item.display_name
      category = {'id': item.id, 'name': name}
      category_index[item.id] = category
  else:
    category_index = None
  return category_index


def run_detect_all():
  # Download and resize all images.
  image_paths = []
  all_image_urls = load_image_json()

  for image_url in all_image_urls:
    image_paths.append(download_and_resize_image(image_url, 1280, 720))

  # Run detection for all modules
  for key in all_modules:
    print(key)
    current_module = all_modules[key]
    module_handle = current_module["module_handle"]
    module_values = current_module["module_values"]
    global detector
    if current_module["signatures"] is not None:
      detector = hub.load(module_handle).signatures['default']
    else:
      detector = hub.load(module_handle)

    #logdir = "test_images_out/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #file_writer = tf.summary.create_file_writer(logdir)

    global PATH_TO_TEST_IMAGES_OUT_DIR
    PATH_TO_TEST_IMAGES_OUT_DIR = os.path.join(
        'test_images_out',
        key,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    
    category_index = load_category_index(module_values["path_to_labels"])

    for image_path in image_paths:
      detect_img(image_path, module_values, category_index)


run_detect_all()
