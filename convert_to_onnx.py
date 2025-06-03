# Codebase to convert the PyTorch Model to the ONNX model
# Reference: https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html#export-the-model-to-onnx-format
# https://docs.pytorch.org/docs/stable/onnx.html

import torch
# Load the trained classifier
from pytorch_model import Classifier, BasicBlock
from PIL import Image

mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
# Load weights
mtailor.load_state_dict(torch.load("./weights/pytorch_model_weights.pth"))
mtailor.eval()

# Load image and preprocess
img = Image.open("./images/n01440764_tench.JPEG")
inp = mtailor.preprocess_numpy(img).unsqueeze(0) 
# res = mtailor.forward(inp)

# Export the model to ONNX format
torch.onnx.export(
    mtailor,                # model to export
    (inp,),                 # inputs of the model,
    "mtailor.onnx",         # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)