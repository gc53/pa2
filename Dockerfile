#ARG ubuntu_version=18.04
#FROM ubuntu:${ubuntu_version}

#Use ubuntu 18:04 as your base image
FROM ubuntu:18.04

#Any label to recognise this image.
LABEL image=Spark-base-image

ENV SPARK_VERSION=3.1.2
ENV HADOOP_VERSION=3.2

#Run the following commands on my Linux machine
#install the below packages on the ubuntu image

RUN apt-get update -qq && \
    apt-get install -qq -y gnupg2 wget openjdk-8-jdk scala

RUN apt-get update && apt-get install -y python3.6 python3-distutils python3-pip python3-apt

#Download the Spark binaries from the repo
WORKDIR /
RUN wget --no-verbose http://www.gtlib.gatech.edu/pub/apache/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz

# Untar the downloaded binaries , move them the folder name spark and add the spark bin on my class path
RUN tar -xzf /spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2 spark && \
    echo "export PATH=$PATH:/spark/bin" >> ~/.bashrc
	
RUN pip3 install numpy
	
#Expose the UI Port 4040
EXPOSE 4040

# Copy the source code and ML Model
COPY . /app

# Create share folder for bind mounts
RUN mkdir /app/share

WORKDIR /app
