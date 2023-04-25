import numpy as np 

with open('img_paths_im0.txt', 'r') as f: 
    img_paths = f.readlines()
    img_paths = [x.strip() for x in img_paths]
np.random.shuffle(img_paths)
print(len(img_paths))

words = ['spoon', 'fork', 'knife', 'banana',
        'pot', 'spatula', 'bowl', 'plate', 'can', 
        'bell pepper', 'cylinder', 'cup', 'orange pot', 
        'chicken', 'brush', 'corn', 'fruit']
# dont segment towels 

import numpy as np
from CaptionGenerator import CaptionGenerator
import os
import dotenv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch 
from utils import * 
import pickle 

print("Loading caption generator")
dotenv.load_dotenv(".env", override=True)
openai_key =  os.getenv("OPENAI_API_KEY")
device = 'cuda:0'

cg = CaptionGenerator(openai_key, device, verbose=True, filter=False, topk=50, gpt4=True)


from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/home/andre/autocaptioning-modelception/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

def load_sa():
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor

predictor = load_sa()


import tqdm 
import os 

segs = {}

root = 'segmented_bridge_objects'
os.makedirs(root, exist_ok=True)

label_counts = {}

for img_path in tqdm.tqdm(img_paths):
    img = Image.open(img_path)
    img = img.resize((512, 512))

    cg.set_words(words)
    count, bboxes, text_labels = cg.get_objects(img)
    if len(bboxes) == 0:
        continue

    img = np.array(img)
    # fname = img_path.split('/')[-1].split('.')[0]

    for bbox, text_label in zip(bboxes, text_labels):
        input_box = np.array(bboxes[0])
        predictor.set_image(img)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        mask = masks[0]
        input_box = input_box.astype(np.int32)

        label = 'bridge_' + text_label.replace(' ', '_')
        os.makedirs(os.path.join(root, label), exist_ok=True)

        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
        fname = f'{label}_{label_counts[label]}'

        np.save(os.path.join(root, label, f'{fname}_npimg.npy'), img)
        np.save(os.path.join(root, label, f'{fname}_npbox.npy'), input_box)
        np.save(os.path.join(root, label, f'{fname}_npmask.npy'), mask)

        # if text_label not in segs:
        #     segs[text_label] = {'imgs': [], 'masks': [], 'boxes': []}
        # segs[text_label]['imgs'].append(img)
        # segs[text_label]['masks'].append(mask)
        # segs[text_label]['boxes'].append(input_box)

# import os
# root = 'segmented_bridge_objects'
# os.makedirs(root, exist_ok=True)
# for label in segs.keys():
#     label = 'bridge_' + label.replace(' ', '_')
#     os.makedirs(os.path.join(root, label), exist_ok=True)

#     images = np.stack(segs[label]['imgs'])
#     masks = np.stack(segs[label]['masks'])
#     boxes = np.stack(segs[label]['boxes'])

#     np.save(os.path.join(root, label, 'npimgs_.npy'), images)
#     np.save(os.path.join(root, label, 'masks_.npy'), masks)
#     np.save(os.path.join(root, label, 'bboxes_.npy'), boxes)








