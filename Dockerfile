FROM "ubuntu:bionic"

RUN apt-get update

RUN apt-get install -y git python3-pip wget

RUN pip3 install numpy scikit-image pillow scipy==1.1.0
RUN pip3 install pyyaml matplotlib cython tensorflow==1.12
RUN pip3 install easydict
RUN pip3 install munkres jupyter
RUN pip3 install scikit-learn pandas catboost
RUN pip3 install sklearn flask

COPY pose_tensorflow/models/mpii /mpii
WORKDIR /mpii
RUN ./download_models.sh

EXPOSE 8080
CMD ["python3", "app.py"]

COPY . /source
WORKDIR /source
RUN rm -rf pose_tensorflow/models/mpii
RUN mv -f /mpii pose_tensorflow/models
