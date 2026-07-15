# The multi-architecture upstream image is pinned by immutable registry digest.
FROM timeloopaccelergy/accelergy-timeloop-infrastructure@sha256:c027f7f57af124ca8500f05f6687ffb0875da4a5b1186934d7daf6a18b143ea2

USER root
COPY requirements-reproduction.txt /tmp/requirements-reproduction.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-reproduction.txt

ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib
WORKDIR /work/optical-loop
ENTRYPOINT []
CMD ["bash"]
