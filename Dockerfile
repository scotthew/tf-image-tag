FROM tensorflow/tensorflow:latest-gpu

#TODO: Set any environment variables
WORKDIR /workspace
COPY ./requirements.txt ./

# Install python dependencies
RUN pip install -r requirements.txt

# Copy font file to container.
COPY ./Roboto-Regular.ttf ./
RUN mkdir -p /usr/share/fonts/truetype/
RUN install -m644 Roboto-Regular.ttf /usr/share/fonts/truetype/

# RUN python -c "import tensorflow as tf;tf.test.gpu_device_name()"

ENTRYPOINT ["tail", "-f", "/dev/null"]