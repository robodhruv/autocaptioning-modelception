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

TODO

This is a bit involved and took me a while to get set up. For now, you can ignore this and just run the `CaptionGenerator` using a server running on Dhruv's Berkeley workstation.
