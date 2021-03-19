# Filename: utils.py
import numpy as np
import cv2

# Implements softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def predictDigit(image,net):
    # Save image
    cv2.imwrite("test.jpg",image)

    # Read image in grayscale mode
    img = cv2.imread("test.jpg",0)

    # Create a 4D blob from image
    blob = cv2.dnn.blobFromImage(img, 1/255, (28, 28))

    # Run a model
    net.setInput(blob)
    out = net.forward()

    # Get a class with a highest score
    out = softmax(out.flatten())
    classId = np.argmax(out)
    confidence = out[classId]

    return classId, confidence
