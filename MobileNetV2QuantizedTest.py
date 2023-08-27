import numpy
import time
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image


# use classes from here "https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a"

# torch.backends.quantized.engine = 'qnnpack'
# model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# model = torch.jit.script(model)

# quanitze doesn't work on my windows environment

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# gets weights for model
weights = models.MobileNet_V2_Weights
model = models.mobilenet_v2(pretrained=True, weights=weights)
model.to(device)

# load classes from txt 
# import json
# with open('imagenet1000_clsidx_to_labels.txt') as f:
    # classes = json.load(f)
# classes = numpy.loadtxt('./imagenet1000_clsidx_to_labels.txt')

# open image
img = Image.open('./testImage3.jpg')
convert = transforms.ToTensor()
img = convert(img)

start_time = time.time()
# start inference loop
with torch.no_grad():
    print("started")
    input = img.to(device)
    input = input.unsqueeze(0)

    output = model(input)
    end_time = time.time()
    print("inference time: ", end_time - start_time)
    print("output: ", output)
    
    top = list(enumerate(output[0].softmax(dim=0)))
    top.sort(key=lambda x: x[1], reverse=True)
    for idx, val in top[:10]:
        print(f"{val.item()*100:.2f}% {classes[idx]}")

# pretty bad accuracy on my test images (roads)