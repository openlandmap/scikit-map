FROM ubuntu:24.04

# Set non-interactive mode for apt-get to prevent it from hanging
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists, install all dependencies, and clean up in a single layer.
# - build-essential: Installs g++, make, and other C++ compiler essentials.
# - cmake: For building your project.
# - libgdal-dev: The development libraries for GDAL (headers, etc.).
# - git: You'll likely need this to check out your code.
# - libgomp1: GNU OpenMP runtime library, to ensure OpenMP support.
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      cmake \
      libgdal-dev \
      git \
      libgomp1 \
      python3 \
      python3-pip \
      python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set a working directory for your project.
WORKDIR /work

# Set the default command to launch a bash shell.
CMD ["/bin/bash"]

