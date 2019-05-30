FROM tensorflow/tensorflow:latest-gpu-py3

COPY datasets/* app/datasets/
COPY neural/*.py neural/Pipfile* /app/neural/
COPY neural/single/*.py /app/neural/single/
COPY neural/multiple/*.py /app/neural/multiple/
COPY setup.py /app

WORKDIR /app/neural

RUN mkdir logs models

WORKDIR /app/neural/models

RUN mkdir single multiple

WORKDIR /app/neural

RUN pip install pipenv
RUN pipenv install --system --deploy

WORKDIR /app

RUN pip install -e .
RUN python -m nltk.downloader stopwords


ENTRYPOINT ["python", "./neural/run.py"]