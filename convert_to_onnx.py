# Codebase to convert the PyTorch Model to the ONNX model
# Reference: https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html#export-the-model-to-onnx-format
# https://docs.pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

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
torch.onnx.export(mtailor,                   # model being run
                  inp,                       # model input (or a tuple for multiple inputs)
                  "mtailor_onnx.onnx",       # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})