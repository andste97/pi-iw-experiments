# Base image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /

ENV WANDB_API_KEY=''

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc ffmpeg libsm6 libxext6 git && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY piiw/ piiw/

RUN pip install -r requirements.txt --no-cache-dir

RUN git clone https://github.com/andste97/gridenvs.git && \
    cd gridenvs && \
    pip install -e . --no-cache-dir

#ENTRYPOINT ["python", "-u", "piiw/online_planning_learning_lightning.py"]
