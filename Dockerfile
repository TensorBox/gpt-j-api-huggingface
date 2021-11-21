FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt install git -y

RUN pip3 install transformers gunicorn uvicorn fastapi

COPY ./*.py /workspace

WORKDIR /workspace

EXPOSE 5000

CMD ["python3", "server.py"]