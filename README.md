# Ultrafast Cardiac Imaging Using Deep Learning
This repo provides CNN model with trained weights in [Open Neural Network Exchange (ONNX)](https://onnx.ai) format and a inference demo script, for the following paper:

[Ultrafast Cardiac Imaging Using Deep Learning For Speckle-Tracking Echocardiography](https://arxiv.org/abs/2306.14265)

## Repo Contents

- [model/](model/model.onnx)model.onnx: model with trained weights
- [data/](data)
  - grid_x.h5: image grid in lateral direction
  - grid_z.h5: image grid in axial direction
  - data_iq_3dw.h5: example IQ data from 3 steered DWs acquisitions along time
  - data_iq_ref.h5: reference IQ data from the compounding of 31 DWs at the frozen time

- [demo.ipynb](demo.ipynb): Running inference 


## Requirements
- [numpy](https://pypi.org/project/numpy)
- [h5py](https://pypi.org/project/h5py)
- [onnxruntime](https://pypi.org/project/onnxruntime)
- [matplotlib](https://pypi.org/project/matplotlib)

## Model inference with the exapmle image data
Code pieces are provided in [jupyter](https://jupyter.org) notebook [demo.ipynb](demo.ipynb)

