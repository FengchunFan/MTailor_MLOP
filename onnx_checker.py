import onnx

onnx_model = onnx.load("mtailor_onnx.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model passed structural check")