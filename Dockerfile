FROM tensorflow/tensorflow:1.14.0-gpu-py3



WORKDIR /code

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y build-essential mpich libpq-dev python3-opencv python3-tk
RUN apt-get install -y wget file

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install

ADD ./requirements.txt /code/
RUN pip install -r requirements.txt
