import base64
import requests
from PIL import Image
from io import BytesIO
from collections import Counter

# def image_to_base64(image: Image.Image) -> str:
#     buffer = BytesIO()
#     image.save(buffer, format="JPEG")
#     return base64.b64encode(buffer.getvalue()).decode()

def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def send_image_to_server(image: Image.Image, caption: str, server_url: str, threshold: float = 0.5) -> dict:
    image_base64 = image_to_base64(image)
    data = {
        'image': image_base64,
        'caption': caption,
        'threshold': threshold,
    }
    response = requests.post(server_url, json=data)
    # print(response)
    # Print the contents of the resonse
    # print(response.content)
    return response.json()

class GLIPClient:
    def __init__(self, server_url: str, threshold: float = 0.5):
        self.server_url = server_url
        self.threshold = threshold

    def send_image(self, image: Image.Image, caption: str) -> dict:
        return send_image_to_server(image, caption, self.server_url, self.threshold)
    
    def tokens_positive_to_words(self, tokens_positive: list, caption: str) -> list:
        words = []
        for token in tokens_positive:
            s = ''
            for subtoken in token:
                s += str(caption[subtoken[0]:subtoken[1]])
            words.append(s)
        return words
    
    def get_object_counts(self, image: Image.Image, words: list) -> dict:
        caption_string = ""
        tokens_positive = []
        seperation_tokens = " . "
        for word in words:
            tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
            caption_string += word
            caption_string += seperation_tokens
        result = self.send_image(image, caption_string)
        tokens = result["result"]["tokens_positive"]
        tokens_text = self.tokens_positive_to_words(tokens, caption_string)
        text_labels = [tokens_text[label - 1] for label in result["result"]["labels"]]
        counts = Counter(text_labels)
        # Get 2dobject centers 
        centers = []
        for bbox in result["result"]["bbox"]:
            centers.append([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        return counts, result["result"]["bbox"], text_labels



if __name__ == "__main__":
    image_path = './test.png'
    # image_path = './GLIP_demo.jpeg'
    # caption = "planter box. blue cabinet. door. window. staircase well "
    # words = ["bedroom", "living room", "bathroom", "kitchen", "dining room", "office room", "gym", "lounge", "laundry room", "closet", "fouyer"]
    words = ["door", "chair", "cabinet", "table", "picture", "cushion", "sofa", "bed", "chest of drawers", "plant", "sink", "toilet", "stool", "towel", "tv monitor", "shower", "bathtub", "counter", "fireplace", "gym equipment", "seating", "clothes", "washing machine", "dishwasher"]

    caption_string = ""
    tokens_positive = []
    seperation_tokens = " . "
    for word in words:
        tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
        caption_string += word
        caption_string += seperation_tokens
    print(caption_string)

    # caption_string = "A green umbrella. A pink striped umbrella. A plain white umbrella."
    # caption_string = "A pink stripped umbrella"

    server_url = "https://9cfc-128-32-255-23.ngrok-free.app/process_image"

    # print(len(words))
    image = Image.open(image_path)
    result = send_image_to_server(image, caption_string, server_url)
    tokens_positive = result["result"]["tokens_positive"]

    print(len(tokens_positive))
    # Print the number of entitiees
    print(result["result"]["entities"])

    # Print the caption_string split up by the ranges specified in tokens_positive
    text_tokens = []
    for token in tokens_positive:
        s = ''
        for subtoken in token:
            s += str(caption_string[subtoken[0]:subtoken[1]])
        text_tokens.append(s)


    # import pdb; pdb.set_trace()

    # print(result["result"]["labels"])
    # print(result["result"]["scores"])


    # Example response 
    # {'bbox': [[0.5800506472587585, 91.60673522949219, 40.15162658691406, 135.87289428710938], 
    # [36.20317077636719, 82.96479797363281, 95.2484359741211, 135.69097900390625], 
    # [43.81270980834961, 94.12615203857422, 65.90650177001953, 128.71340942382812], 
    # [65.03579711914062, 94.86456298828125, 85.56037139892578, 129.58103942871094]], 
    # 'labels': [2, 1, 3, 3], 
    # 'scores': [0.7338275909423828, 0.7121511101722717, 0.6166518330574036, 0.5360745787620544]}
    #     
    # Print the names of the unique labels
    # print(set([captions[label - 1] for label in result["result"]["labels"]])) 
    # 

    # Remove any labels that are out of range
    # labels = [label for label in result["result"]["labels"] if label <= len(words)]



    # Print the names with the quanitty of each
    from collections import Counter
    entities = result["result"]["entities"]
    bbox = result["result"]["bbox"]
    # Print the lables
    print(result["result"]["labels"])

    # If the bbox shares 90% of the area with another bbox, then combine the labels
    # for i in range(len(bbox)):
    #     for j in range(len(bbox)):
    #         if i == j:
    #             continue
    #         # Get the area of the intersection
    #         intersection_area = (min(bbox[i][2], bbox[j][2]) - max(bbox[i][0], bbox[j][0])) * (min(bbox[i][3], bbox[j][3]) - max(bbox[i][1], bbox[j][1]))
    #         # Get the area of the union
    #         union_area = (bbox[i][2] - bbox[i][0]) * (bbox[i][3] - bbox[i][1]) + (bbox[j][2] - bbox[j][0]) * (bbox[j][3] - bbox[j][1]) - intersection_area
    #         # Get the intersection over union
    #         if union_area == 0:
    #             continue
    #         iou = intersection_area / union_area
    #         if iou > 0.9:
    #             # Combine the labels
    #             result["result"]["labels"][i] = result["result"]["labels"][j]
    #             # Remove the bbox
    #             result["result"]["bbox"][j] = [0, 0, 0, 0]
    #             # Remove the score
    #             result["result"]["scores"][j] = 0


    # print(Counter([text_tokens[label - 1] for label in result["result"]["labels"]]))

    # Print a list of each object with the center location of the bounding box and the label
    for i in range(len(result["result"]["bbox"])):
        center = [(result["result"]["bbox"][i][0] + result["result"]["bbox"][i][2]) / 2, (result["result"]["bbox"][i][1] + result["result"]["bbox"][i][3]) / 2]
        # Round the center to 1 decimal place
        center = [round(center[0], 1), round(center[1], 1)]
        print(center, text_tokens[result["result"]["labels"][i] - 1])

    

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    for bbox in result["result"]["bbox"]:
        # if bbox == [0, 0, 0, 0]:
        #     continue
        draw.rectangle(bbox, outline="red")
        label = result["result"]["labels"][result["result"]["bbox"].index(bbox)] - 1
        print(label)
        print(len(entities))
        # if label >= len(words):
        #     continue
        # draw.text((bbox[0], bbox[1]), captions[label], fill="red")
        # Make the text larger
        # Set the font and size you want
        font_size = 30
        # font = ImageFont.load_default()
        # import pdb; pdb.set_trace()

        # ImageDraw.textbbox
        # Use the font in your draw.text() function
        print(text_tokens[label])
        draw.text((bbox[0], bbox[1]), text_tokens[label], fill="red")    # Save the image
        
    if image.mode == 'RGBA':
        image = image.convert('RGB')
        
    image.save("result.jpg")

