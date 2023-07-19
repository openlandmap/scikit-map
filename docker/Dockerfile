# Use an official GDAL image as the base image
FROM osgeo/gdal:3.2.0

# install pip
RUN apt-get update && apt-get -y install python3-pip git --fix-missing

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt /app/

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    git clone https://github.com/scikit-map/scikit-map.git && \
    cd scikit-map && \
    python setup.py install


ENTRYPOINT ["tail", "-f", "/dev/null"]

