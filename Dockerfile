FROM python:3.11-slim

WORKDIR /app

# libgomp is required by LightGBM on Linux
RUN apt-get update -qq && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy pandas scikit-learn lightgbm pyyaml joblib \
    fastapi uvicorn[standard]

CMD ["python", "-m", "blue.run_blue"]
