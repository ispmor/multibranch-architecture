FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER puszkarski.bartosz@gmail.com

## DO NOT EDIT the 3 lines.
RUN mkdir /phd
COPY ./ /phd
WORKDIR /phd

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip3 install -r requirements.txt
