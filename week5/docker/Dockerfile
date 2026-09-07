# PyTorch 2.14 на CUDA 12.6: достаточно новый для текущих transformers/peft.
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime

WORKDIR /workspace

# PYTHONPATH: scripts/*.py кладут в sys.path папку scripts/, а не корень
#   проекта, без этого "from app import ..." не работает.
# HF_HOME: кэш моделей и датасетов, монтируется как named volume, чтобы
#   скачанное переживало перезапуск контейнера.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    HF_HOME=/root/.cache/huggingface

# Сначала зависимости: слой кэшируется и пересобирается только при изменении файла.
# --break-system-packages: образ использует системный Python Debian, помеченный
# как externally managed (PEP 668), и обычный pip install отказывается работать.
# В контейнере защищать нечего — torch лежит в тех же site-packages.
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# В dev-compose эти две папки перекрываются bind-mount'ом, правки на хосте
# работают без пересборки. В prod запускается именно эта копия.
COPY app ./app
COPY scripts ./scripts

CMD ["bash"]
