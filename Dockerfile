FROM python:3.8-slim

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH to the /app directory
ENV PYTHONPATH=/app

# Expose the port FastAPI will run on
EXPOSE 80

COPY . /app

# Command to run the FastAPI app
# CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "$PORT"]


