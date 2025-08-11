FROM python:3.10.14

ARG PORT=8888
ENV PORT=$PORT

EXPOSE $PORT


WORKDIR /app
COPY . .

RUN pip install pipenv
RUN pip install torch


RUN pipenv install --system --deploy


# The --dev flag tells pipenv to install both [packages] and [dev-packages].
RUN pipenv install --system --deploy --dev

CMD ["python","main.py"]