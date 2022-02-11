FROM tensorflow/tensorflow:latest-gpu


#TODO: Set any environment variables
WORKDIR /workspace
RUN pip install "tensorflow>=2.0.0"
RUN pip install --upgrade tensorflow-hub
RUN pip install "matplotlib"
RUN pip install "tensorboard"

COPY ./Roboto-Regular.ttf ./
RUN mkdir -p /usr/share/fonts/truetype/
RUN install -m644 Roboto-Regular.ttf /usr/share/fonts/truetype/

# RUN python -c "import tensorflow as tf;tf.test.gpu_device_name()"

ENTRYPOINT ["tail", "-f", "/dev/null"]