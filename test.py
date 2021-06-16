from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net

import cv2
import torch.nn
import torchvision
import numpy as np
from PIL import Image


use_gpu = False
device = torch.device('cpu')
if use_gpu:
    device = torch.device('cuda:0')

model_test = U_Net(3, 1)
model_test.load_state_dict(torch.load('Unet_epoch_100_batchsize_8.pth', map_location=device))

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((640, 1280)),
    torchvision.transforms.ToTensor(),
])

test_image = 'test/1.jpg'
image = Image.open(test_image)
image_transformed = data_transform(image)


model_test.to(device)
image_transformed.to(device)

pred_tb = model_test(image_transformed.unsqueeze(0))
pred_tb = torch.sigmoid(pred_tb)
pred_tb = pred_tb.cpu().data.numpy()[0]
pred_tb = np.squeeze(pred_tb)
pred_tb = pred_tb * 255.0
# pred_tb = np.clip(pred_tb, 0, 255)
pred_tb = np.asarray(pred_tb, dtype=np.uint8)

image_name = 'output.png'
cv2.imwrite(image_name, pred_tb)





