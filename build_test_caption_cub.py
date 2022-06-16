import random
from image_synthesis.data.cub200_dataset import Cub200Dataset


def oversample(dataset, num_samples):
    dataset_len = len(dataset)
    out = [dataset[random.randint(0, dataset_len-1)] for idx in range(num_samples)]
    return out
    
    

def main():
    dataset = Cub200Dataset('data/cub', phase='test')
    caption_dict = dataset.caption_dict
    captions = []
    for v in caption_dict.values():
        captions += v

    print(len(captions))
        
    selected = oversample(captions, 30000)

    with open('cub_test_captions.txt', 'w') as f:
        for caption in selected:
            f.write(caption.strip() + '\n')
            
main()