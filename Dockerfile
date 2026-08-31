FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -qq && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy pandas scikit-learn lightgbm pyyaml joblib \
    fastapi "uvicorn[standard]"

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8080

# Set RELOAD=1 to enable hot-reload (development mode)
# e.g. docker run -e RELOAD=1 -v $(pwd):/app raptor
ENTRYPOINT ["./entrypoint.sh"]
