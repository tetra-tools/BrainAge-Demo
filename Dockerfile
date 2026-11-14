# Brain age inference image (CPU).
#
#   docker build -t brainage:0.1.0 .                                  # model mounted at runtime
#   docker build -t brainage:0.1.0-sfcn1.0.0 --target bundled \
#       --build-arg BUNDLE_DIR=dist/brainage-sfcn-1.0.0 .             # model baked into the image
#
# See README.md for usage.

ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm

# --------------------------------------------------------------------------- build
FROM ${BASE_IMAGE} AS build
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CPU-only torch (needed by torchio); ~10x smaller than the default CUDA wheel
RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN pip install --no-deps /src

# ---------------------------------------------------------- runtime (shared base)
FROM ${BASE_IMAGE} AS runtime-base
LABEL org.opencontainers.image.title="brainage" \
      org.opencontainers.image.description="Brain age from raw T1w MRI: N4 + rigid/affine MNI registration + SFCN (ONNX)" \
      org.opencontainers.image.source="https://github.com/tetra-tools/BrainAge-Demo" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 brainage \
    && mkdir -p /opt/brainage/model /data /out && chown brainage /data /out
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    BRAINAGE_BUNDLE=/opt/brainage/model \
    MPLCONFIGDIR=/tmp
USER brainage
WORKDIR /data
ENTRYPOINT ["brainage"]
CMD ["--help"]

# ------------------------------------------- target "bundled": model baked in
FROM runtime-base AS bundled
ARG BUNDLE_DIR=model
COPY --chown=root:root ${BUNDLE_DIR}/ /opt/brainage/model/
RUN brainage info --bundle /opt/brainage/model

# ------------------- target "runtime" (default, last stage): model mounted at run time
FROM runtime-base AS runtime
