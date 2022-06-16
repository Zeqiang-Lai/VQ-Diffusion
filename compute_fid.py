import torch_fidelity

from torchlight.data.dataset import ImageFolder
from torchlight.transforms.functional import crop_center, hwc2chw, chw2hwc
import cv2
import torch

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

class Dataset(ImageFolder):
    def __init__(self, root, mode='color'):
        super().__init__(root, mode)
        
    def __getitem__(self, index):
        img, path = super().__getitem__(index)
        img = chw2hwc(img)
        h,w,c = img.shape
        if h < 256:
            img = image_resize(img, height=256)
        h,w,c = img.shape
        if w < 256:
            img = image_resize(img, width=256)
        img = crop_center(img, 256, 256)
        h,w,c = img.shape
        assert h >= 256 and w>=256, f'h{h}, w{w}'
        img = hwc2chw(img)
        img = (img * 255).astype('uint8')
        img = torch.tensor(img)
        return img

metrics_dict = torch_fidelity.calculate_metrics(
    input1=Dataset('data/cub/images'),
    input2=Dataset('RESULT_cub200_pretrained'),
    cuda=True,
    fid=True,
    verbose=True,
)

print(metrics_dict)