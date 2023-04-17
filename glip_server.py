import requests
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request
import os
# pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# Start the docker container with nvidia-docker run -it -v $(pwd):/workspace glipserver

# Use this command to evaluate the GLPT-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# Conver the above wget into somethign that can be run in the script


if not os.path.exists('MODEL'):
    os.makedirs('MODEL')
    url = 'https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth'
    output_path = 'MODEL/glip_large_model.pth'
    urllib.request.urlretrieve(url, output_path)

config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

# result, _ = glip_demo.run_on_web_image(image, caption, 0.5)


import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = base64.b64decode(data['image'])
    pil_image = Image.open(BytesIO(image_data))
    caption = data['caption']
    threshold = data.get('threshold', 0.5)

    # Convert image to tensor
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    result, top_predictions = glip_demo.run_on_web_image(image, caption, threshold)
    print(type(result), type(thing))
    print(result)
    print(thing)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)