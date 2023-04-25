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


if __name__ == "__main__":
    parent_dir = '/home/andre/autocaptioning-modelception/source_images/bridge_data'

    for root, dirs, files in os.walk(parent_dir):
        obj_name = root.split('/')[-1].replace('_', ' ')
        npimgs_, bboxes_, masks_ = [], [], []
        print('Processing ', obj_name)
        for file in files:
            if file.endswith('.jpg'):
                image = get_img(os.path.join(root, file))
                image = image.resize((512, 512))

                cg.set_words([obj_name])
                count, bboxes, text_labels = cg.get_objects(image)
                if len(bboxes) == 0:
                    continue
                
                print('Detected object for ', obj_name)
                input_box = np.array(bboxes[0])
                img = np.array(image)
                predictor.set_image(img)
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                mask = masks[0]
                input_box = input_box.astype(np.int32)

                npimgs_.append(img)
                bboxes_.append(input_box)
                masks_.append(mask)

        if len(npimgs_) == 0:
            continue
        np.save(os.path.join(root, 'npimgs_.npy'), np.stack(npimgs_))
        np.save(os.path.join(root, 'bboxes_.npy'), np.stack(bboxes_))
        np.save(os.path.join(root, 'masks_.npy'), np.stack(masks_))


        # save the results
        # # save_path = image_path.split('/')[] + '%s.npy'
        # np.save(save_path % 'mask', mask)
        # np.save(save_path % 'bbox', input_box)
        # np.save(save_path % 'npimg', img)


        

# if __name__ == "__main__":
#     image_paths = os.listdir('./images/')
#     image_paths = [path for path in image_paths if path.endswith('.jpg')]
#     image_paths = [os.path.join('./images/', path) for path in image_paths]

#     # obj_names = []
#     # obj_bboxes = []
#     # obj_masks = []
#     # obj_images = []
    
#     for image_path in image_paths:
#         obj_name = image_path.split('/')[-1].split('.')[0]
#         file_name = obj_name
#         # remove numbers
#         obj_name = ''.join([i for i in obj_name if not i.isdigit()])
#         obj_name = ' '.join(obj_name.split('_'))
#         print(obj_name)

#         image = get_img(image_path)
#         # crop out center square
#         w, h = image.size
#         if w > h:
#             image = image.crop((w/2 - h/2, 0, w/2 + h/2, h))
#         else:
#             image = image.crop((0, h/2 - w/2, w, h/2 + w/2))
#         image = image.resize((512, 512))
        
#         cg.set_words([obj_name])
#         count, bboxes, text_labels = cg.get_objects(image)
#         if len(bboxes) == 0:
#             continue

#         input_box = np.array(bboxes[0])
#         img = np.array(image)
#         predictor.set_image(img)
#         masks, _, _ = predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=input_box[None, :],
#             multimask_output=False,
#         )
#         mask = masks[0]

#         # save the results
#         np.save(f'./images/{file_name}.npy', obj_name)
#         np.save(f'./images/{file_name}_bbox.npy', input_box)
#         np.save(f'./images/{file_name}_mask.npy', mask)
#         np.save(f'./images/{file_name}_image.npy', image)
    


#         # obj_names.append(obj_name)
#         # obj_bboxes.append(input_box)
#         # obj_masks.append(mask)
#         # obj_images.append(image)
    
#     # # save the results
#     # np.save('obj_names.npy', obj_names)
#     # np.save('obj_bboxes.npy', obj_bboxes)
#     # np.save('obj_masks.npy', obj_masks)
#     # np.save('obj_images.npy', obj_images)

        