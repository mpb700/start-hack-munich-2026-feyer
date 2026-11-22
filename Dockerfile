# Linux environment for `python benchmark.py --cpus 1` (CPU affinity needs Linux).
# benchmark.py is the harness without JAX; the supplied one is benchmark_original.py
# (`docker run --entrypoint python <image> benchmark_original.py --cpus 1 --output /out/x.json`).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first so code changes do not invalidate this layer.
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install -e . && mkdir -p /out

# Extra arguments are passed to benchmark.py, e.g. `--repeats 3 --save-arrays /out/x.npz`.
ENTRYPOINT ["python", "benchmark.py", "--cpus", "1", "--output", "/out/benchmark-docker.json"]
CMD []
