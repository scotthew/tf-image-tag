# tf-image-tag

TF Image Tag

## Docker

[cuda wsl user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
[TF Docker](https://www.tensorflow.org/install/docker)

`docker compose up --build -d`

Verify tensorflow installation

```bash
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

TF Jupyter

```bash
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-jupyter
```

Verify nvidia

```bash
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-sm
```

## TF Image Classification

[TF Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
[Top 4 Pre-Trained Models](https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/)
[TF2 Models](https://tfhub.dev/tensorflow/collections/object_detection/1)
[TF object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
[TF2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)