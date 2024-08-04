# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy specific files and directories into the container
COPY app.py /app/
COPY models/model/best_model.h5 /app/models/model
COPY models/transformers /app/models/transformers
COPY static/ /app/static/
COPY templates/ /app/templates/

COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 9000

# Set the entry point to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]