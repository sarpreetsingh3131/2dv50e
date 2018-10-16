FROM python:3.6
RUN apt-get update
RUN mkdir -p /opt/machine-learner
WORKDIR /opt/machine-learner
ADD . /opt/machine-learner
RUN pip3 install -r requirements.txt
EXPOSE 8000
CMD python3 manage.py runserver 0.0.0.0:8000
