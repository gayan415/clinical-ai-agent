FROM python:3.11-slim

WORKDIR /app

# Install only model serving dependencies (no PyTorch, no LangChain, no ChromaDB)
# Keeps the image small — this container ONLY serves predictions
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn>=0.29.0 \
    xgboost>=2.0.0 \
    scikit-learn>=1.4.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    joblib>=1.3.0 \
    pydantic>=2.0.0

# Copy only the model code and artifacts needed for serving
COPY model/__init__.py model/__init__.py
COPY model/predict.py model/predict.py
COPY model/features.py model/features.py

# Model artifacts will be mounted or copied at deploy time
# For local testing: docker run -v ./models:/app/models ...
RUN mkdir -p models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "model.predict:app", "--host", "0.0.0.0", "--port", "8000"]
