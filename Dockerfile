FROM python:3.12-slim

WORKDIR /app

# Upgrade pip and setuptools
# RUN pip install --upgrade pip setuptools wheel

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH to the /app directory
ENV PYTHONPATH=/app

# Expose the port FastAPI will run on
EXPOSE 80

# Command to run the FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]
