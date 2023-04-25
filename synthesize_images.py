from absl import app, flags, logging
import os
import imgaug as ia
import imgaug.augmenters as iaa
import re
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import cv2
import tqdm 

FLAGS = flags.FLAGS
flags.DEFINE_string('src', '/home/andre/autocaptioning-modelception/source_images/bridge_data/', 'path to source images')
flags.DEFINE_string('arr', '/home/andre/autocaptioning-modelception/gpt4_captions_2/', 'path to arrangement captions and tikz')
flags.DEFINE_integer('num', 1000, 'number of images to synthesize')
flags.DEFINE_string('out', '/home/andre/autocaptioning-modelception/synthesized_images/', 'path to output directory')

OBJECT_KEYS = ['bell_pepper', 'brush', 'carrot', 
                'corn', 'grapes', 'knife', 'salt_shaker', 
                'spatula', 'spoon']
CONTAINER_KEYS = ['metal_pot', 'plate', 'orange_pot']

def augment_image(rng, image):
    pass 

def parse_tikz_commands(tikz_commands):
    tuples = []
    for command in tikz_commands:
        # extract the x and y coordinates
        xy = re.search(r'\((.*),(.*)\)', command)
        x = float(xy.group(1))
        y = float(xy.group(2))
        
        # extract the label
        label = re.search(r'node\{(.*?)\}', command)
        if label:
            label = label.group(1)
        else:
            # if no label is found, use the shape as the label
            shape = re.search(r'(\w+)$', command)
            label = shape.group(1)
        
        tuples.append((x, y, label))
        
    return tuples

def get_crop(img, bbox):
    crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    return crop 

def paste_array(A, B, B_mask, x, y):
    """
    Paste 2D NumPy array B into another array A, centered at integer coordinates x, y.
    Handle the case where B can go over the edges of A.
    """
    A = A.copy()

    s = max(B.shape[0], B.shape[1])
    A = np.pad(A, s, mode='constant', constant_values=0)[:, :, s:-s]
    x, y = x + s, y + s

    B_mask = B_mask[:, :, np.newaxis]
    A[x-B.shape[0]//2:x+B.shape[0]//2 + B.shape[0]%2, 
        y-B.shape[1]//2:y+B.shape[1]//2 + B.shape[1] % 2] *= 1 - B_mask
    A[x-B.shape[0]//2:x+B.shape[0]//2 + B.shape[0]%2, 
        y-B.shape[1]//2:y+B.shape[1]//2 + B.shape[1] % 2] += B * B_mask
    A = A[s:-s, s:-s]

    return A

def paste_mask(A, B, x, y):
    """
    Paste 2D NumPy array B into another array A, centered at integer coordinates x, y.
    Handle the case where B can go over the edges of A.
    """
    A = A.copy()

    s = max(B.shape[0], B.shape[1])
    A = np.pad(A, s, mode='constant', constant_values=0)
    x, y = x + s, y + s

    A[x-B.shape[0]//2:x+B.shape[0]//2 + B.shape[0]%2, 
        y-B.shape[1]//2:y+B.shape[1]//2 + B.shape[1] % 2] += B
    A = A[s:-s, s:-s]
    A = np.clip(A, 0, 1)

    return A

def bilinear_interpolation(x, y, q11, q12, q22, q21):
    return q11 * (1 - x) * (1 - y) + \
           q21 * x * (1 - y) + \
           q12 * (1 - x) * y + \
           q22 * x * y

def main(_):
    fs = os.listdir(FLAGS.arr)
    caption_paths = [f for f in fs if 'captions' in f]
    tikz_paths = [f.replace('captions', 'tikzs') for f in caption_paths]

    # print(caption_paths)

    # read captions and tikzs
    captions = []
    tikzs = []
    for caption_path, tikz_path in zip(caption_paths, tikz_paths):
        with open(os.path.join(FLAGS.arr, caption_path), 'r') as f:
            cs = [c.strip() for c in f.readlines()]
            captions.extend(cs)
        with open(os.path.join(FLAGS.arr, tikz_path), 'r') as f:
            ts = [t.strip() for t in f.readlines()]
            tikzs.extend(ts)

    arrangements = list(zip(captions, tikzs))

    # read source images
    src_images = {}
    for root, subdirs, files in os.walk(FLAGS.src):
        obj_name = root.split('/')[-1]
        if not 'npimgs_.npy' in files:
            continue
        try:
            npimgs = np.load(os.path.join(root, 'npimgs_.npy'))
            bboxes = np.load(os.path.join(root, 'bboxes_.npy'))
            masks = np.load(os.path.join(root, 'masks_.npy'))
            
        except:
            continue 
        src_images[obj_name] = (npimgs, bboxes, masks)


    bg_image_dir = os.path.join(FLAGS.src, 'background')
    bg_images = []
    for root, subdirs, files in os.walk(bg_image_dir):
        for f in files:
            if f.endswith('.jpg'):
                bg = Image.open(os.path.join(root, f))
                bg = bg.resize((512, 512))
                bg = np.array(bg)
                bg_images.append(bg)
                
    grid_size = (6, 6)

    captions = []
    os.makedirs(FLAGS.out, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.out, 'images'), exist_ok=True)

    for nidx in tqdm.tqdm(range(FLAGS.num)):
        caption, tikz = arrangements[nidx % len(arrangements)]
            
        objects = [k for k in OBJECT_KEYS]
        containers = [k for k in CONTAINER_KEYS]
        np.random.shuffle(objects)
        np.random.shuffle(containers)

        object_symbols = ['A', 'B', 'C', 'D']
        container_symbols = ['m', 'n']

        for obj, sym in zip(objects, object_symbols):
            # caption_sym = '<%s>' % sym
            caption_sym = sym 
            # caption = caption.replace(caption_sym, 'the ' + obj.replace('_', ' '))
            caption = re.sub(r'\b'+sym+'(?=[\s,.])', 'the ' + obj.replace('_', ' '), caption)
            tikz_sym = '{%s}' % sym
            tikz = tikz.replace(tikz_sym, '{%s}' % obj)
        
        for obj, sym in zip(containers, container_symbols):
            caption_sym = sym
            caption = re.sub(r'\b'+sym+'(?=[\s,.])', 'the ' + obj.replace('_', ' '), caption)
            tikz_sym = '{%s}' % sym
            tikz = tikz.replace(tikz_sym, '{%s}' % obj)

        caption = caption.replace('container ', '')
        caption = caption.replace('object ', '')
        caption = caption.replace('the the', 'the')
        
        # print('[CAPTION]')
        # print(caption)
        # print('[TIKZ]')
        # print(tikz)

        try:
            state = parse_tikz_commands(tikz.split(';')[:-1])
        except:
            print('unparsable tikz')
            continue
        
        try: 
            state = sorted(state, key=lambda x: -(x[0] - 0)**2 - (x[1] - 0)**2)

            bg = np.random.choice(np.arange(len(bg_images)))
            bg = bg_images[bg]

            image = bg.copy()

            vertices = np.array([[100, 512-100], [100, 100], [512-100, 100], [512-100, 512-100]])
            vertices = vertices + np.random.randint(-50, 50, size=vertices.shape)
            vertices = vertices.clip(0, 512)

            # TODO not used now
            inpainting_mask = np.zeros((512, 512))
            kernel = np.ones((3,3),np.uint8)

            for i, j, obj_name in state:
                objs, bboxs, masks = src_images[obj_name]
                idx = np.random.choice(np.arange(len(objs)))
                obj = objs[idx].astype(np.uint8)
                mask = masks[idx].astype(np.uint8)
                bbox = bboxs[idx]

                # dilated_mask = cv2.dilate(mask, kernel, iterations=5)
                # obj_edge = dilated_mask - mask
                
                mask_crop = get_crop(mask, bbox)
                # obj_edge_crop = get_crop(obj_edge, bbox)
                obj_crop = get_crop(obj, bbox)
                
                # print(i/grid_size[0], j/grid_size[1], *vertices)
                cx, cy = bilinear_interpolation(i/grid_size[0], j/grid_size[1], *vertices)
                cx, cy = int(cx), int(cy)
                image = paste_array(image, obj_crop, mask_crop, cy, cx)

                # inpainting_mask = paste_mask(inpainting_mask, obj_edge_crop, cy, cx)
        except: 
            print('error, probably invalid gpt4 generation')
            continue

        image = Image.fromarray(image)
        image.save(os.path.join(FLAGS.out, f'images/IMG_{int(nidx):06d}.jpg'))
        captions.append(str(nidx) + ': ' + caption)

    with open(os.path.join(FLAGS.out, 'captions.txt'), 'w') as f:
        f.write('\n'.join(captions))

if __name__ == '__main__':
    app.run(main)