## Inference test with OpenVINO Inference Engine(IE)


from PIL import Image
import numpy as np
import platform
import os


# Check path  exists in `PYTHONPATH`.
is_win = 'windows' in platform.platform().lower()

""" 
# OpenVINO 2018.
if is_win:
    message = "Please run `C:\\Intel\\computer_vision_sdk\\bin\\setupvars.bat` before running this."
else:
    message = "Add the following line to ~/.bashrc and re-run.\nsource ~/intel/computer_vision_sdk/bin/setupvars.sh"
"""

# OpenVINO 2019.
if is_win:
    message = 'Please run "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat" before running this.'
else:
    message = "Add the following line to ~/.bashrc and re-run.\nsource /opt/intel/openvino/bin/setupvars.sh"

assert 'computer_vision_sdk' in os.environ['PYTHONPATH'] or 'openvino' in os.environ['PYTHONPATH'], message

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IEPlugin
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

def pre_process_image(imagePath, img_height=224):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    image = Image.open(imagePath)
    processedImg = image.resize((h, w), resample=Image.BILINEAR)

    # Normalize to keep data between 0 - 1
    processedImg = (np.array(processedImg) - 0) / 255.0

    # Change data layout from HWC to CHW
    processedImg = processedImg.transpose((2, 0, 1))
    processedImg = processedImg.reshape((n, c, h, w))

    return image, processedImg, imagePath

# Plugin initialization for specified device and load extensions library if specified.
plugin_dir = None
model_xml = './model/frozen_model.xml'
model_bin = './model/frozen_model.bin'
# Devices: GPU (intel), CPU, MYRIAD
plugin = IEPlugin("MYRIAD", plugin_dirs=plugin_dir)
# Read IR
net = IENetwork(model=model_xml, weights=model_bin)
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
# Load network to the plugin
exec_net = plugin.load(network=net)
del net

# Run inference
fileName = './data/elephant.jpg'
image, processedImg, imagePath = pre_process_image(fileName)
res = exec_net.infer(inputs={input_blob: processedImg})
# Access the results and get the index of the highest confidence score
output_node_name = list(res.keys())[0]
res = res[output_node_name]
idx = np.argsort(res[0])[-1]
idx

from tensorflow.keras.applications.inception_v3 import decode_predictions
print('Predicted:', decode_predictions(res, top=3)[0])

import time
times = []
for i in range(20):
    start_time = time.time()
    res = exec_net.infer(inputs={input_blob: processedImg})
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1/mean_delta
print('average(sec):{},fps:{}'.format(mean_delta,fps))