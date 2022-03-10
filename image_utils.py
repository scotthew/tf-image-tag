import os
import json

from PIL import Image

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

def load_image_json():
  image_file_path = os.getenv('IMAGE_FILE_PATH')
  if image_file_path is not None:
    print("loading image from file path: ", image_file_path)
    with open(image_file_path, 'r') as fh:
      image_urls_json = json.load(fh)
    
    return image_urls + image_urls_json['images']
  else:
    return image_urls


def save_image(path, image_path, image_np):
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)

    # Convert to image and write to out file
    out_image_path = os.path.join(path, os.path.basename(image_path))
    print("Out Image Name: %s" % (out_image_path))
    out_img = Image.fromarray(image_np)
    out_img.save(out_image_path, "jpeg")
    return

def save_fig(path, image_path, ax):
    out_image_path = os.path.join(path, os.path.basename(image_path))
    print("Out Image Name: %s" % (out_image_path))
    ax.savefig(out_image_path, format='png', dpi=300)
    return
