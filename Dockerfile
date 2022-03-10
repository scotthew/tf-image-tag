FROM tensorflow/tensorflow:latest-gpu

#TODO: Set any environment variables
WORKDIR /workspace
COPY ./requirements.txt ./

#https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
#RUN apt-get update
#RUN apt-get install -y python3-opencv
#RUN apt-get install libgl1

# Install python dependencies
RUN pip install -r requirements.txt

# Copy font file to container.
COPY ./Roboto-Regular.ttf ./
RUN mkdir -p /usr/share/fonts/truetype/
RUN install -m644 Roboto-Regular.ttf /usr/share/fonts/truetype/

# RUN python -c "import tensorflow as tf;tf.test.gpu_device_name()"

ENTRYPOINT ["tail", "-f", "/dev/null"]