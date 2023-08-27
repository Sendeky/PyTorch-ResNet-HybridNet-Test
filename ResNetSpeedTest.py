import time
import torch
import torchvision

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img = read_image("./testImage3.jpg")
# img = Image.open("./testImage3.jpg")
# convert_tensor = torchvision.transforms.ToTensor()
# img = convert_tensor(img)
# img = img.type(torch.uint8)
# img.unsqueeze(0)

# Step 1: Initialize model with the best available weights
# weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.5)
model.eval()

#Step 2: Initialize the inference transforms
preprocess = weights.transforms()
preprocess

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
start_time = time.time()
prediction = model(batch)[0]
end_time1 = time.time()
print("inference time:", end_time1 - start_time)
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30,)
im = to_pil_image(box.detach())
end_time2 = time.time()
print("bbox time: ", end_time2 - start_time)
im.show()

# Ryzen 5 2600 (3.4GHz, 6 core) inefernce time: ~0.25s