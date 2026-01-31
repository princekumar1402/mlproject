FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose the port Flask runs on
EXPOSE 8080

# Start the Flask app
CMD ["python", "application.py"]
