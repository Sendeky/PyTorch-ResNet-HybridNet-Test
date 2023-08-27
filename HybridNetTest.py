import torch
import torchvision
# import numpy as np

from torchvision import transforms
from PIL import Image
from torchvision.utils import draw_bounding_boxes


def convertImage(img):
    convert_tensor = transforms.ToTensor()  # converts image to PyTorch Tensor
    resize         = transforms.Resize((640, 384))  # image to model needs to be  680x384

    img = convert_tensor(img)
    img = resize(img)

    return img

# load model
model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)

#inference
img = Image.open('./testImage.jpg').convert('RGB')
image = convertImage(img)
img = image.unsqueeze(0)                   # need to add dimension to tensor (model expects 4D Tensor while image is 3D)
print(img.shape)

# img = torch.randn(1,3,640,384)
features, regression, classification, anchors, segmentation = model(img)
print(anchors)
a     = anchors.type(torch.uint8)
a     = torchvision.ops.box_convert(boxes=a, in_fmt='xywh', out_fmt='xyxy')
print('a:', a)
# torchvision.ops.box_convert('xywh')
image = image.type(torch.uint8)

# draw_bounding_boxes(image=image, boxes=a) # requires (x min, y min, x max, y min)