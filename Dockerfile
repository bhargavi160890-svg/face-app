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

# Download models at build time from HuggingFace (files present in the repo are NOT overwritten).
# Render env vars:
#   HF_MODEL_BASE = https://huggingface.co/UBHT/face-siamese/resolve/main
#   MODEL_VERSION = v1, v2, ...  -> change it after uploading new models (forces fresh download, bypasses Docker cache)
# Files not uploaded yet (e.g. expression_model.t7 before Stage 4) are skipped, build continues.
ARG HF_MODEL_BASE=""
ARG SIAMESE_URL=""
ARG MODEL_VERSION="v1"
RUN echo "Model version: $MODEL_VERSION" && \
    if [ -n "$SIAMESE_URL" ] && [ ! -f app/Hackathon_setup/siamese_model.t7 ]; then \
      wget -q -O app/Hackathon_setup/siamese_model.t7 "$SIAMESE_URL"; fi && \
    if [ -n "$HF_MODEL_BASE" ]; then \
      for f in siamese_model.t7 face_classifier.sav expression_model.t7; do \
        if [ ! -f "app/Hackathon_setup/$f" ]; then \
          (wget -q -O "/tmp/$f" "$HF_MODEL_BASE/$f" && mv "/tmp/$f" "app/Hackathon_setup/$f" && echo "downloaded $f") \
          || echo "skip $f (not on HuggingFace yet)"; \
        fi; \
      done; \
    fi

# Render/AWS inject PORT; default 8001 for local Docker
ENV PORT=8001
# MEMORY OPTION 2: limit glibc memory arenas + single-threaded math libs
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
EXPOSE 8001

CMD ["python", "app/main.py"]