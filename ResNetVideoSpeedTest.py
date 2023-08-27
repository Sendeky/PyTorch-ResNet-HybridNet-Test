import cv2
import time
import torch
import torchvision
import numpy

from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.io.video import read_video
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cap = cv2.VideoCapture("./myTest.mp4")
cap = read_video("./myTest.mp4")
# cap = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

# Step 1: Initialize model with the best available weights
# weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.5)
model.eval()

#Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
# batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
# start_time = time.time()
# prediction = model(batch)[0]
# end_time1 = time.time()
# print("inference time:", end_time1 - start_time)
# labels = [weights.meta["categories"][i] for i in prediction["labels"]]
# box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                        #   labels=labels,
                        #   colors="red",
                        #   width=4, font_size=30,)
# im = to_pil_image(box.detach())
# end_time2 = time.time()
# print("bbox time: ", end_time2 - start_time)
# im.show()

while cap.isOpened():
    try:
        # read frame
        ret, frame = cap.read()
        # frame = Image.open(frame).convert('RGB')
    except:
        continue

    if ret:
        # do inference
        
        # convert = transforms.ToTensor()
        # frame = convert(frame)
        # frame = frame.unsqueeze(0)
        print("f shape: ", frame.shape)
        frame = torch.from_numpy(frame)     # cv2 gives us a numpy array
        print("f2 shape: ", frame.shape)
        frame = preprocess(frame)
        print("f3 shape: ", frame.shape)
        frame = frame.unsqueeze(0)
        print("f4 shape: ", frame.shape)

        start_time = time.time()

        prediction = model(frame)
        print("pred: ", prediction)
        
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(frame, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30,)
        im = to_pil_image(box.detach())
        end_time2 = time.time()
        print("bbox time: ", end_time2 - start_time)
        im.show()
        print("yay")