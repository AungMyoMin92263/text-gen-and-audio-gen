FROM python:3.10.14



EXPOSE $PORT


WORKDIR /app
COPY . .

RUN pip install pipenv

RUN pipenv install --system --deploy


CMD ["python","main.py"]