import random
from image_synthesis.data.cub200_dataset import Cub200Dataset

dataset = Cub200Dataset('data/cub', phase='test')
caption_dict = dataset.caption_dict
captions = []
for v in caption_dict.values():
    captions += v

print(len(captions))

selected = random.sample(captions, 30000//8)

with open('cub_test_captions.txt', 'w') as f:
    for caption in selected:
        f.write(caption.strip() + '\n')