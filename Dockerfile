FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mare_baselines ./mare_baselines

# /data: the dataset (read-only), /results: metrics, checkpoints and example images
ENTRYPOINT ["python", "-m"]
CMD ["mare_baselines.seg", "--help"]
