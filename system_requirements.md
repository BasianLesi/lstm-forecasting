## Install tensorflow on raspberry pi

raspberry os debian buster: https://downloads.raspberrypi.org/raspios_lite_armhf/images/ \
python 3.7.3
Tensorflow - 2.4.0: https://github.com/lhelontra/tensorflow-on-arm/releases

#### install prerequisites
```python
pip install cython
sudo apt-get install gcc python3.7-dev
pip install pycocotools==2.0.0
pip install --upgrade numpy==1.20.1
```
##### in this file:
sudo nano /home/pi/tensorflow/lib/python3.7/site-packages/tensorflow/python/ops/array_ops.py

https://github.com/tensorflow/models/issues/9706#:~:text=edited-,Hi%20%40aniketbote%20I%20posted%20this%20answer%20in%20Stack%20Overflow%3A%20https,from%20tensorflow.math%20import%20reduce_prod,-%F0%9F%91%8D


##### we change the following:
```python
from tensorflow.math import reduce_prod\
def _constant_if_small(value, shape, dtype, name):\
  try:\
    if np.prod(shape) < 1000: ## --> it to if reduce_prod(shape) < 1000:\
      return constant(value, shape=shape, dtype=dtype, name=name)\
  except TypeError:\
    # Happens when shape is a Tensor, list with Tensor elements, etc.\
    pass\
  return None
```

##### reinstall h5py
```python
sudo apt-get update\
pip uninstall h5py\
pip list\
sudo apt-get install libhdf5-dev\
pip install h5py\
pip install --upgrade h5py==2.10.0
```
