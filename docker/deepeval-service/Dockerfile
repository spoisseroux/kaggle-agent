FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    deepeval \
    fastapi \
    uvicorn[standard] \
    pydantic

# Copy application
COPY app.py /app/

# Expose port
EXPOSE 8001

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
