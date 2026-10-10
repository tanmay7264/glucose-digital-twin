FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy scikit-learn
COPY twin ./twin
COPY web ./web
COPY out ./out
COPY run.py .
CMD ["python", "run.py", "--serve"]
