from PIL import Image
from os import listdir

import os

classname = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

for items in classname:
    os.mkdir('ResizeEmotion/Validating/' + items)

for classN in classname:
    path = f'Emotion/Validating/{classN}'
    namelist = listdir(path)
    for name in namelist:
        img = Image.open(f'Emotion/Validating/{classN}/{name}') # image extension *.png,*.jpg
        new_width  = 128
        new_height = 128
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(f'ResizeEmotion/Validating/{classN}/{name}')