FROM tensorflow/tensorflow:2.2.2-gpu-py3

WORKDIR /code

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y build-essential mpich libpq-dev python3-opencv python3-tk
RUN apt-get install -y wget file vim cmake libopenmpi-dev zlib1g-dev

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install

ADD ./requirements.txt /code/
RUN pip install -r requirements.txt

ADD openai-baselines /openai-baselines
RUN apt-get install -y git libcublas-dev
RUN sed -i 's/10.1/10.2/' /etc/ld.so.conf.d/cuda-10-1.conf && \
    cp /usr/local/cuda-10.2/targets/x86_64-linux/include/* /usr/include && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcublas.so /usr/lib/ && \
    ldconfig

RUN cd /openai-baselines && \
    pip install --upgrade pip && \
    pip install -e .[mpi,tests,docs] && \
    pip install git+https://github.com/cudamat/cudamat.git && \
    rm -rf $HOME/.cache/pip
