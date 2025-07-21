
# Python 3.13 try:
```
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv bin/micromamba .local/bin/
eval "$(micromamba shell hook --shell bash)"
micromamba activate arcov2
sudo apt-get remove python3-pybind11
micromamba install pybind11
python setup.py build_ext --inplace
```

"$(python3.13-config --ldflags --embed)"

It doesn't work - kernel dies on sb.readData

# Python 3.8 try:

eval "$(micromamba shell hook --shell bash)"
micromamba activate arcov2_38
python setup_cmake.py build_ext --inplace

# On server nibble
# 