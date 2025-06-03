# Containing following Classes and Functionality
#   Onnx Model loading and prediction call
#   Pre-processing of the Image [Sample code provided in pytorch_model.py]

# Reference: 
import onnx

class Preprecessor:
    def __init__(self):
        pass

class ONNX_Model:
    # Load Onnx model
    def __init__(self):
        self.onnx_model = onnx.load("mtailor.onnx")
        onnx.checker.check_model(self.onnx_model)

    # Input an pre-processed image
    # Output a class label
    def predict(self) -> int:
        
        return label