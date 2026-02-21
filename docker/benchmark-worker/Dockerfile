# docker/benchmark-worker/Dockerfile
#
# CPU benchmark worker image.
# Based on apache/spark:3.5.5-python3 (Java 17 + Spark 3.5.5 + Python 3.11).
# Includes all non-GPU benchmark frameworks: polars, duckdb, pandas, dask, pyspark.
#
# Build:
#   docker build -t <hub>/hsm-bench-worker:latest docker/benchmark-worker/
# Push:
#   docker push <hub>/hsm-bench-worker:latest

FROM apache/spark:3.5.5-python3

USER root

# System deps for numpy/scipy native libs (already in base, but ensure curl is present)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python benchmark deps
RUN pip install --no-cache-dir \
    polars \
    duckdb \
    pyarrow \
    pandas \
    "dask[dataframe]" \
    boto3 \
    s3fs \
    psutil \
    pynvml \
    huggingface_hub

# Copy repo into /app
COPY . /app
WORKDIR /app

# Add repo root + Spark python + py4j to PYTHONPATH
ENV PYTHONPATH=/app:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip

# Ensure /home/spark exists (DuckDB httpfs extension needs a writable home directory)
RUN mkdir -p /home/spark && chown spark:spark /home/spark

# hadoop-aws + aws-sdk JARs for S3A support (must match Spark 3.5.5 hadoop version)
# Spark 3.5.5 ships with Hadoop 3.3.4
RUN curl -fSL -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
        "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar" && \
    curl -fSL -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar \
        "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar"

# Switch back to spark user (required by Spark k8s driver)
USER spark

ENTRYPOINT ["python3", "/app/benchmarks/run_single.py"]
