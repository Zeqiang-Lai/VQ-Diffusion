from image_synthesis.data.cub200_dataset import Cub200Dataset

config = {
    "target": "image_synthesis.data.utils.image_preprocessor.DalleTransformerPreprocessor",
    "params": {
              "size": 256,
              "phase": "val"
    }

}
dataset = Cub200Dataset('../data/cub', phase='test', im_preprocessor_config=config)
data = dataset.__getitem__(0)
print(data.keys())
print(data['image'].shape)
print(data['text'])

# dict_keys(['image', 'text'])
# (3, 256, 256)
# this bird is white with grey and has a long, pointy beak.

from torchlight.utils.io import imsave

imsave('test.png', data['image'].astype('uint8').transpose(1, 2, 0))