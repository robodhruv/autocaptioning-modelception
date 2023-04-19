# Autocaptioning with Modelception

This notebook shows a quick walkthrough of the caption generation pipeline.

1. (region proposal) Run GLIP to detect relevant objects and get crops using the bounding boxes
2. (caption per region) Use BLIP2 to generate captions for each crop
3. (filtering) Get the CLIP similarities between every caption and every crop and choose the best caption for each crop
4. (filtering) Throw away the captions that do not have the highest cosine similarities with their crop
5. (summarization) Ask GPT4 to remove any outliers
6. (summarization) Ask GPT4 to generate a final caption for the entire image



The repository is self-contained. To use, follow these steps:
- In a virtual env, `pip install -r requirements.txt`
- Set the [OpenAI API key](https://platform.openai.com/account/api-keys), `export OPENAI_API_KEY=<your-key-here>`


To play around with the pipeline and benchmark, please see `playground.ipynb`.

To see the GLIP model in action, simply run
```
python glip_client.py
```

To use the CaptionGenerator outside of this notebook, simply run
```
python CaptionGenerator.py
```



### GLIP_server

#### Option 1 (Setup server)

GLIP has some weird dependencies, so I'd recommended running a server inside a docker image. Follow these steps:
1. Make sure your workstation has `nvidia-docker`. If not, install this first.
2. `docker pull pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`
3. Install the [GLIP repository](https://github.com/microsoft/GLIP) and `cd` inside it. 
4. Enter the docker container using `nvidia-docker run -it -v $(pwd):/workspace pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`. Then run these steps:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers 
python setup.py build develop --user
```
5. You are now ready to run the server. Copy the `glip_server.py` file from this repo into the GLIP repo.
6. Run `python glip_server.py`. This will take a while to download the right model and if successful, give you a server address. This will likely be `172.17.0.2:5000`.
7. To ensure this works fine, edit `glip_client.py` in this repo to use the correct server address (e.g., `172.17.0.2:5000/process_image`), then run `python glip_client.py`. If successful, this will output some bbox and caption predictions of the test image.

#### Option 2

If this is too involved and you only want to play around with this briefly, feel free to run the `CaptionGenerator` using a server running on Dhruv's Berkeley workstation (thestral.bair.berkeley.edu:1027/process_image), and follow step 7 above.
