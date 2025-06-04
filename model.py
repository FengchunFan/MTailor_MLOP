# Containing following Classes and Functionality
#   Onnx Model loading and prediction call
#   Pre-processing of the Image [Sample code provided in pytorch_model.py]

# Reference: https://docs.pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

import onnxruntime
from torchvision import transforms
import numpy as np
from PIL import Image

# Reference to Preprocess function in Classifier class in pytorch_model 
# Convert img to numpy array to ONNX Model
# Input should be img not path
class Preprocessor:
    def __init__(self):
        self.resize = transforms.Resize((224, 224))   #must same as here
        self.crop = transforms.CenterCrop((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Take in image directly, easier to fit with deployed application design
    def fit(self, img: Image.Image) -> np.ndarray:
        img = self.resize(img)
        img = self.crop(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        img = img.unsqueeze(0)
        return img.numpy()

class ONNX_Model:
    # Run the model with ONNX Runtime
    def __init__(self):
        # Create an inference session for the model 
        self.ort_session = onnxruntime.InferenceSession("mtailor_onnx.onnx", providers=["CPUExecutionProvider"])

    # Input an pre-processed image (numpy array)
    # Output a class label
    def predict(self, input_array: np.ndarray) -> int:
        # Maps 'input': input_array
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_array}
        # List of possibility for all classes
        ort_outs = self.ort_session.run(None, ort_inputs)
        # Give label to the index class with highest probability
        label = int(np.argmax(ort_outs[0]))
        return label