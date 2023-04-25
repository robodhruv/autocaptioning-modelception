### query GPT-4 for arrangements and natural language captions
from absl import app, flags, logging
import os 

FLAGS=flags.FLAGS
# flags.DEFINE_string('objects', 'object_list.txt', 'Path to object list')
# flags.DEFINE_string('containers', 'container_list.txt', 'Path to container list')
flags.DEFINE_integer('num', 1000, 'Number of arrangements to generate')
flags.DEFINE_integer('id', 0, 'Run id')
flags.DEFINE_string('output', 'arrangements/', 'Output directory')

STATE_PROMPT = """Suppose you have a set of objects and containers. 
You have containers n, m and objects A, B, C. 
Come up with ways to arrange the objects and containers on a tabletop. 
Describe your arrangement with words and illustrate them with a simple tikz figure, where each item is represented as a circle with a text label. 
Imagine that the table space is a 6 by 6 grid. 
Consider (0, 0) to be the bottom left corner, and (6, 0) to be the bottom right corner.
Only use integer coordinates. If you want to put an item in a container, draw them at the same coordinate.
Some examples: 

[CAPTION] 
Arrange A, B, C, and D into a square.
[TIKZ] 
`\draw (1, 4) circle [radius=0.5] node{A};
\draw (4, 4) circle [radius=0.5] node{B};
\draw (1, 1) circle [radius=0.5] node{C};
\draw (4, 1) circle [radius=0.5] node{D};`. 

[CAPTION] 
Put A in n.
[TIKZ] 
`\draw (2, 2) circle [radius=0.5] node{n};
\draw (2, 2) circle [radius=0.5] node{A};`. 

[CAPTION]
Put A to the left of B and put C in m.
[TIKZ]
`\draw (1, 2) circle [radius=0.5] node{A};
\draw (4, 2) circle [radius=0.5] node{B};
\draw (3, 3) circle [radius=0.5] node{m};
\draw (3, 3) circle [radius=0.5] node{C};`

Generate 5 different arrangements using objects: %s and containers: %s.
"""


# TASK_PROMPT = """Suppose you have a set of objects and containers. You have containers a, b and objects A, B, C. 
# Come up with ways to arrange the objects and containers on a tabletop. You may use all or none of the objects or containers. 
# Illustrate the arrangment with a simple tikz figure, where each item is represented as a circle with a text label. 
# Then, come up with a task where you move a single object. Describe the task with words and output the coordinates of the object after the task is completed.
# For the tikz figures, imagine that the table space is a 6 by 6 grid. 
# Consider (0, 0) to be the bottom left corner, and (6, 0) to be the bottom right corner.
# Only use integer coordinates. If you want to put an item in a container, draw them at the same coordinate.
# You can use spatial relations like "move A to the left of B" to describe the movement, 
# as well as more complex relations like "between", "towards", or abstract shapes like "in a line" or "in a square". Be creative. 
# Move only a single object, and do not move containers.
# Some examples: 

# [INITIAL]
# `\draw (1, 4) circle [radius=0.5] node{A};
# \draw (4, 4) circle [radius=0.5] node{B};
# \draw (1, 1) circle [radius=0.5] node{C};
# \draw (1, 3) circle [radius=0.5] node{D};`
# [TASK]
# Arrange A, B, C, and D into a square
# [FINAL]
# `\draw (1, 4) circle [radius=0.5] node{A};
# \draw (4, 4) circle [radius=0.5] node{B};
# \draw (1, 1) circle [radius=0.5] node{C};
# \draw (4, 1) circle [radius=0.5] node{D};`
# [MOVE] 
# `D (1, 3) -> (4, 1)`

# [INITIAL]
# `\draw (4, 4) circle [radius=0.5] node{a};
# \draw (2, 2) circle [radius=0.5] node{A};`. 
# [TASK] 
# Put A in a: 
# [FINAL]
# `\draw (4, 4) circle [radius=0.5] node{a};
# \draw (4, 4) circle [radius=0.5] node{A};`. 
# [MOVE] 
# `A (2, 2) -> (4, 4)`. 

# [INITIAL]
# `\draw (5, 2) circle [radius=0.5] node{A};
# \draw (4, 2) circle [radius=0.5] node{B};
# \draw (2, 1) circle [radius=0.5] node{b};
# \draw (3, 3) circle [radius=0.5] node{C};`
# [TASK]
# Put A to the left of B 
# [FINAL]
# `\draw (1, 2) circle [radius=0.5] node{A};
# \draw (4, 2) circle [radius=0.5] node{B};
# \draw (2, 1) circle [radius=0.5] node{b};
# \draw (3, 3) circle [radius=0.5] node{C};`
# [MOVE]
# `A (5, 2) -> (1, 2)`

# [INITIAL]
# `\draw (1, 1) circle [radius=0.5] node{A};
# \draw (1, 5) circle [radius=0.5] node{B};
# \draw (5, 5) circle [radius=0.5] node{C};`
# [TASK]
# Move A toward the center of the table
# [FINAL]
# `\draw (2, 2) circle [radius=0.5] node{A};
# \draw (1, 5) circle [radius=0.5] node{B};
# \draw (5, 5) circle [radius=0.5] node{C};`
# [MOVE]
# `A (1, 1) -> (2, 2)`
# """




import openai
import numpy as np 

# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#             {"role": "system", "content": PROMPT},
#         ]
# )

# num_objects_distr = [0, 1/3, 1/3, 1/3]
num_objects_distr = [0, 0.48, 0.48, 0.04]
# num_objects_distr = [1/3, 1/3, 1/3]
num_containers_distr = [0.5, 0.45, 0.05]

def main(_):
    print('started process ', FLAGS.id)
    captions, tikzs = [], []
    for _ in range(FLAGS.num):
        num_objects = np.random.choice([0, 1, 2, 3], p=num_objects_distr)
        num_containers = np.random.choice([0, 1, 2], p=num_containers_distr)
        objects = ['A', 'B', 'C', 'D'][:num_objects]
        containers = ['m', 'n'][:num_containers]
        prompt = STATE_PROMPT % (', '.join(objects), ', '.join(containers))

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                    {"role": "system", "content": prompt},
                ]
        )
        print('got response, parsing')
        content = response['choices'][0]['message']['content']

        try: 
            for i in range(5):
                example = content.split('[CAPTION]')[1+i]
                caption, tikz = example.split('[TIKZ]')
                caption = caption.replace('\n', '')
                caption = caption.strip()
                tikz = tikz.replace('\n', '')
                print('[CAPTION]', caption)
                print('[TIKZ]', tikz)   
                captions.append(caption)
                tikzs.append(tikz)
        except: 
            print('failed to parse')
            continue

    # write to file
    os.makedirs(FLAGS.output, exist_ok=True)
    with open(os.path.join(FLAGS.output, f'captions_{FLAGS.id}.txt'), 'w') as f:
        f.write('\n'.join(captions))
    with open(os.path.join(FLAGS.output, f'tikzs_{FLAGS.id}.txt'), 'w') as f:
        f.write('\n'.join(tikzs))




if __name__ == '__main__':
    app.run(main)