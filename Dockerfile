#Base image being pytorch w/ cuda 
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

#Metadata on container
LABEL maintainer="Noire Meyers & Kristopher Jimenez Poston"
LABEL description="Socio-Technical Audit Environment for YOLOv8"

#Syst Dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

#  Set the working directory in the container (app)
WORKDIR /app

#  Copy dependencies first for better caching (improved caching)
COPY requirements.txt .

#  Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#  Copy the rest of the application code
COPY . .

#Output should be going to terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Container starts within a shell
CMD ["python", "tests/test_environment.py"]