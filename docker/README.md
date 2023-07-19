# README.md

## Docker and Docker-Compose

This document provides instructions on how to use the provided `docker-compose.yml` and Docker directly for the `scikit-map` service.

### Prerequisites

- Docker: Please ensure Docker is installed on your machine. You can follow the installation instructions based on your operating system from the official Docker docs: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

- Docker Compose: Docker Compose is also required. If you're on Windows or Mac and installed Docker Desktop, Docker Compose should already be included. For Linux users, please follow these instructions to install Docker Compose: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

### Using Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications. Here's how to use it for our `scikit-map` service:

1. **Clone the repository**
    ```
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Build and Run the Docker Compose**
    ```
    docker-compose up --build
    ```
    This command builds the Docker image as per the Dockerfile instructions and then starts the container. The `-d` flag can be used to start the container in detached mode (in the background).

3. **Stop the Docker Compose**
    ```
    docker-compose down
    ```
    This command stops and removes the container. Any changes made inside the container that are not saved to a volume will be lost.

### Using Docker Directly

You can also run the Docker container directly using Docker commands. Follow these steps:

1. **Clone the repository**
    ```
    git clone https://github.com/scikit-map/scikit-map.git
    cd scikit-map/docker
    ```

2. **Build the Docker Image**
    ```
    docker build -t scikit-map -f Dockerfile .
    ```
    This command builds a Docker image named "scikit-map" using the Dockerfile in the current directory.

3. **Run the Docker Container**
    ```
    docker run -d --name scikit-map -v "$(pwd):/app" scikit-map
    ```
    This command runs the "scikit-map" container in the background (`-d`), names it "scikit-map" (`--name`), and mounts the current directory (`$(pwd)`) to `/app` in the container (`-v`).

4. **Stop and Remove the Docker Container**
    ```
    docker stop scikit-map
    docker rm scikit-map
    ```
    These commands stop and remove the "scikit-map" container. Any changes made inside the container that are not saved to a volume will be lost. 

### Important Note

Please note that the `scikit-map` service uses the current directory (`.`) as a volume, mounted at `/app` in the container. Any changes you make inside the `/app` directory in the container will be reflected in your host machine's current directory, and vice versa.
