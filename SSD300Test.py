import time
import torch
import torchvision

from PIL import Image


ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ssd_model.to('cuda')
ssd_model.eval()

# opens image
image = Image.open('/testImage.png').convert("RGB")

convert_tensor = torchvision.transforms.ToTensor()  # converts image to PyTorch Tensor
resize         = torchvision.transforms.Resize((300,300))  # image to model needs to be  680x384

image          = convert_tensor(image).to(device)
img            = resize(image).to(device)
img            = img.unsqueeze(0)
img.to(device)

start_time = time.time()    # inference time start
with torch.no_grad():
    detections_batch = ssd_model(img)
    end_time = time.time()                          # inference time ends
    print("inference time: ", end_time-start_time)  # prints inference time


# get boxes only above 10% confidence
results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.10) for results in results_per_input]

# gets COCO classes so it can identify object
classes_to_labels = utils.get_coco_object_dictionary()


## shows the image with the bounding boxes
from matplotlib import pyplot as plt
import matplotlib.patches as patches

for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)  # tensor is in wrong order, need to permute it so that plt can show it
    plt.imshow(img.cpu())
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
plt.show()

