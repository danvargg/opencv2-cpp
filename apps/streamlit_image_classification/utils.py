# Filename: utils.py

import numpy as np
import onnxruntime
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models

def read_image(image_path):
    image = Image.open(image_path)
    return image

def convert_model():
    # Use a pre-trained model from Torchvision
    model = models.resnet50(pretrained=True)

    # Create some sample input in the shape this model expects
    dummy_input = torch.randn(1, 3, 224, 224)

    # It's optional to label the input and output layers
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    # Use the exporter from torch to convert to onnx 
    # model (that has the weights and net arch)
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)

def transform_image(image):
    """ This function transforms an image
    in order to crop it to pre-defined size
    and then normalized. """

    transform = transforms.Compose([            #[1]
     transforms.Resize(256),                    #[2]
     transforms.CenterCrop(224),                #[3]
     transforms.ToTensor(),                     #[4]
     transforms.Normalize(                      #[5]
     mean=[0.485, 0.456, 0.406],                #[6]
     std=[0.229, 0.224, 0.225]                  #[7]
     )])

    return transform(image)

def model_inference(model,img_pil):
    """ This function carries out model
    inference using the ONNX model path
    and image as input."""
    # Transform image
    img_t = transform_image(img_pil)
    batch_t = torch.unsqueeze(img_t, 0)

    # Create ONNX runtime session for model
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # Carry out model inference
    result = session.run([output_name], {input_name: batch_t.numpy()})[0]

    # Get class names
    with open('imagenet_classes.txt') as f:
      classes = [line.strip() for line in f.readlines()]

    # Get class name
    class_name = classes[np.argmax(result.ravel())]

    return class_name