# First stage: install dependencies
FROM python:3.10-slim AS build

WORKDIR /summarizer_api

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install uvicorn
RUN pip install uvicorn

# Copy the application code and client_secrets.json file
COPY app /summarizer_api/app
COPY client_secrets.json /summarizer_api/client_secrets.json

# Set the environment variable for Google Application Credentials
#ENV GOOGLE_APPLICATION_CREDENTIALS="/summarizer_api/client_secrets.json"

# Download the model files
RUN python app/download.py

# Second stage: create a clean image
FROM python:3.10-slim

WORKDIR /summarizer_api

# Copy the installed dependencies from the build stage
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /usr/local/bin /usr/local/bin

# Copy the application code and required files
COPY --from=build /summarizer_api /summarizer_api

# Expose port 8080
EXPOSE 8080

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
