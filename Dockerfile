# Use a light version of Python
FROM python:3.10-slim

# Set the folder inside the container
WORKDIR /app

# Install system dependencies for Postgres (psycopg2)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy your requirements first (better for speed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Tell Render which port we are using
EXPOSE 10000

# Start the application
CMD ["python", "app.py"]