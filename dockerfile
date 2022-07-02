# FROM --platform=linux/x86_64 python:3.8
# FROM --platform=linux/amd64 python:3.8
FROM armswdev/tensorflow-arm-neoverse:r22.06-tf-2.9.1-eigen as base
USER root
EXPOSE 8000


WORKDIR /usr/src/app
ADD . . 

RUN apt-get update -qq && \
    apt-get install -yqq apt-utils && \
    apt-get install -yqq procps && \
    apt-get install -yqq gcc &&\
    apt-get install -yqq make &&\
    apt-get install -yqq --no-install-recommends vim &&\
    
    #Clean-up
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean


RUN pip install --upgrade pip
RUN pip install pipenv


RUN pipenv install --system --deploy --ignore-pipfile
CMD  ["/bin/sh", "-ec", "sleep infinity"]



# docker image prune -f
# docker container prune -f
