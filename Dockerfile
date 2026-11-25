# syntax=docker/dockerfile:1
FROM python:3.14.6-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
WORKDIR /build
COPY requirements.runtime.lock /build/requirements.runtime.lock
RUN python -m pip install --no-cache-dir --no-compile --target /app/runtime/algorithm-deps \
    -r requirements.runtime.lock
RUN PYTHONPATH=/app/runtime/algorithm-deps python -m pip check
COPY src /build/src
COPY scripts /build/scripts
COPY configs /build/configs
COPY runtime/compact /build/runtime/compact
COPY artifacts/models/ppocr-v5 /build/artifacts/models/ppocr-v5
COPY docker /build/docker
RUN python /build/docker/prepare_runtime.py /build /app

FROM base AS runtime
WORKDIR /app
COPY --from=build /app/runtime/algorithm-deps /app/runtime/algorithm-deps
COPY --from=build /app/src /app/src
COPY --from=build /app/scripts /app/scripts
COPY --from=build /app/configs /app/configs
COPY --from=build /app/runtime/compact /app/runtime/compact
COPY --from=build /app/runtime/font /app/runtime/font
COPY --from=build /app/artifacts/models /app/artifacts/models
COPY --from=build /app/docker /app/docker
ENV PYTHONPATH=/app/src:/app/scripts:/app/runtime/algorithm-deps:/app/docker \
    CAD_SERVICE_STORAGE=/app/artifacts/profile-service-v1/storage CAD_DEMO_OCR_MODELS=/app/artifacts/models/ppocr-v5 \
    CAD_SERVICE_TIMEOUT_SECONDS=1800 WINDIR=/app/runtime/font \
    XDG_CACHE_HOME=/app/artifacts/profile-service-v1/cache
RUN groupadd --gid 10001 cad && useradd --uid 10001 --gid cad --no-create-home cad \
    && mkdir -p /app/artifacts/profile-service-v1/storage /app/artifacts/profile-service-v1/cache \
       /app/artifacts/.cache /app/artifacts/service-runtime-v1/deps /app/artifacts/cv-skeleton-runtime-v1/deps \
    && chown -R cad:cad /app/artifacts/profile-service-v1 /app/artifacts/.cache
USER cad
RUN python /app/docker/service.py check
EXPOSE 8000
ENTRYPOINT ["python", "/app/docker/service.py"]
CMD ["web"]
