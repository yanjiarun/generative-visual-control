FROM nvcr.io/nvidia/pytorch:25.06-py3-igpu

ENV CUDA_MODULE_LOADING=LAZY
ENV PYTHONUNBUFFERED=1

RUN pip3 install --no-cache-dir diffusers pillow onnx

WORKDIR /workspace

CMD ["/bin/bash"]


