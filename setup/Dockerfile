FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "usat_designer/parameter_sampling/generate_random_params.py", "--num", "1", "--config", "usat_designer/parameter_sampling/5OA_7_0_4.yaml", "--bucket_name", "usat_decodings_3"]