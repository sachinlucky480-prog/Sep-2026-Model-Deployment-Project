# Use python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application and serialized model into the container
COPY app.py .
COPY superkart_model.joblib .

# Expose port 5000
EXPOSE 5000

# Run the Flask app using Gunicorn or python directly
CMD ["python", "app.py"]
