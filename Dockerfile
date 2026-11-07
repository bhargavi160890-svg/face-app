# Slim base image = smaller/faster builds on Render & AWS
FROM python:3.11-slim-bookworm

COPY requirements.txt requirements.txt

# Only the system libs OpenCV-headless needs (removed g++, graphviz, openmpi, hdf5 etc. - not used)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 myuser

USER myuser

COPY --chown=myuser app app

# Optional: download the model at build time if it is NOT in the repo (GitHub 100MB limit).
# Set SIAMESE_URL as a build arg / Render env var, e.g. https://huggingface.co/<user>/<repo>/resolve/main/siamese_model.t7
ARG SIAMESE_URL=""
RUN if [ -n "$SIAMESE_URL" ] && [ ! -f app/Hackathon_setup/siamese_model.t7 ]; then \
      wget -q -O app/Hackathon_setup/siamese_model.t7 "$SIAMESE_URL"; fi

# Render/AWS inject PORT; default 8001 for local Docker
ENV PORT=8001
EXPOSE 8001

CMD ["python", "app/main.py"]
