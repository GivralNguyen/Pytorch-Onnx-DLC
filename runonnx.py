import numpy as np 
import torch 
from vision.utils import box_utils
import cv2 
import torch.nn.functional as F
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import os
import onnxruntime
from vision.ssd.data_preprocessing import PredictionTransform
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print(scores.shape)
prob_threshold = 0.4 
iou_threshold = 0.45
top_k = 10
candidate_size = 200
image_path = "/media/HDD/ssdlite/pytorch-ssd/human.jpg"
label_path = "/media/HDD/ssdlite/pytorch-ssd/models_right/voc-model-labels.txt"

class_names = [name.strip() for name in open(label_path).readlines()]

net = create_mobilenetv1_ssd(len(class_names), is_test=True)


orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape
images = np.fromfile("/media/HDD/ssdlite/pytorch-ssd/human.raw",dtype=np.float32).reshape(1,3, 300, 300)

# scores = np.fromfile("scores.raw",dtype=np.float32).reshape(1,3000,21)
# boxes = np.fromfile("boxes.raw",dtype=np.float32).reshape(1,3000,4)

# scores = torch.from_numpy(scores)
# boxes = torch.from_numpy(boxes)

# confidences_dlc = np.fromfile("/media/HDD/snpe-1.52.0.2724/ssd_mobilenet_v2_coco_2018_03_29/output/Result_0/confidences.raw",dtype=np.float32).reshape(1,3000,21)
# locations_dlc = np.fromfile("/media/HDD/snpe-1.52.0.2724/ssd_mobilenet_v2_coco_2018_03_29/output/Result_0/locations.raw",dtype=np.float32).reshape(1,3000,4)

# confidences = np.fromfile("/media/HDD/ssdlite/pytorch-ssd/confidences.raw",dtype=np.float32).reshape(1,3000,21)
# locations = np.fromfile("/media/HDD/ssdlite/pytorch-ssd/locations.raw",dtype=np.float32).reshape(1,3000,4)

session = onnxruntime.InferenceSession("/media/HDD/ssdlite/pytorch-ssd/models/mb1-ssd-sim.onnx", None)
input_name = session.get_inputs()[0].name
output_name_0 = session.get_outputs()[0].name
output_name_1 = session.get_outputs()[1].name
#print(input_name)
#print(output_name)
 
confidences,locations = session.run([output_name_0, output_name_1], {input_name: images})


confidences = torch.from_numpy(confidences)
locations = torch.from_numpy(locations)

scores = F.softmax(confidences, dim=2)
boxes = box_utils.convert_locations_to_boxes(
    locations, net.priors, net.config.center_variance, net.config.size_variance
)
boxes = box_utils.center_form_to_corner_form(boxes)

boxes = boxes[0]
scores = scores[0]

cpu_device = torch.device("cpu")
boxes = boxes.to(cpu_device)
scores = scores.to(cpu_device)

picked_box_probs = []
picked_labels = []
for class_index in range(1, scores.size(1)):
    probs = scores[:, class_index]
    mask = probs > 0.3
    probs = probs[mask]
    if probs.size(0) == 0:
        continue
    subset_boxes = boxes[mask, :]
    box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
    box_probs = box_utils.nms(box_probs, None,
                              score_threshold=prob_threshold,
                              iou_threshold=iou_threshold,
                              sigma=0.5,
                              top_k=top_k,
                              candidate_size=candidate_size)
    picked_box_probs.append(box_probs)
    picked_labels.extend([class_index] * box_probs.size(0))

picked_box_probs = torch.cat(picked_box_probs)
picked_box_probs[:, 0] *= width
picked_box_probs[:, 1] *= height
picked_box_probs[:, 2] *= width
picked_box_probs[:, 3] *= height

boxes, labels, probs = picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0] + 20), int(box[1] + 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
