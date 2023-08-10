FROM python:3.11.4

RUN apt-get update

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ui.py"]