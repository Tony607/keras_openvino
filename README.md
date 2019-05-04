# [How to run Keras model inference x3 times faster with CPU and Intel OpenVINO](https://www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/) | DLology Blog


## System requirement

- 6th-8th Generation Intel® Core™
- Intel® Xeon® v5 family
- Intel® Xeon® v6 family
### Operating Systems

- Ubuntu* 16.04.3 long-term support (LTS), 64-bit
- CentOS* 7.4, 64-bit
- Windows* 10, 64-bit

## How to Run
Require [Python 3.5+](https://www.python.org/ftp/python/3.6.4/python-3.6.4.exe) and [Jupyter notebook](https://jupyter.readthedocs.io/en/latest/install.html) installed
### Clone or download this repo
```
git clone https://github.com/Tony607/keras_openvino
```
### Install OpenVINO
Setup OpenVINO on your machine, choose your OS on [this page](https://software.intel.com/en-us/openvino-toolkit/choose-download), follow the instruction to download and install it.

### Install required libraries
`pip3 install -r requirements.txt`


Run the `setupvars.bat` before calling `jupyter notebook`.
```
"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```
Or in Linux
add the following line to `~/.bashrc`
```
source /opt/intel/openvino/bin/setupvars.sh
```
In a terminal run,
```
jupyter notebook
```

In the opened browser window open
```
keras-openvino-ImageNet.ipynb
```

### Run Inference on NCS (Neural compute stick 1 or 2)
Once you have the inference model, plugin the NCS, run this script.
```
python3 ncs_inference.py
```