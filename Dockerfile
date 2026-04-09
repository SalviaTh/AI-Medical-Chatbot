# Use official lightweight Python image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default port (Render will configure its own PORT env variable, but we default to 8080)
EXPOSE 8080

# Use gunicorn as the production WSGI server
# Start application using $PORT if available, else 8080
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} app:app --workers 2 --timeout 120
