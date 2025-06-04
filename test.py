# Codebase to test the code/model written

# Load the classes we implemented
from model import Preprocessor, ONNX_Model
from PIL import Image

# Get list of test cases (imgs)
# Currently we have two valid ones
# Model should return error on invalid image path
# Format: (image_path, image_label)
test_cases = [
    ("./images/n01440764_tench.JPEG", 0),           # Valid case
    ("./images/n01440764_tench_2.JPEG", 0),         # Super-highlighted fish
    ("./images/n01667114_mud_turtle_2.JPEG", 35),   # Corped the turtle img to different shape
    ("./images/noth_even_exist.JPEG", 99),          # None existing image, should not prevent test case 3 from happening
    ("./images/n01667114_mud_turtle.JPEG", 35)      # Valid case
]

# Call Preprocessor and Model
Preproc = Preprocessor()
Model = ONNX_Model()

# Run through test_cases
for i, (image_path, image_label) in enumerate(test_cases):
    # Check if file exists, output will be in runner code
    try: 
        img = Image.open(image_path)
        inp = Preproc.fit(img)
        predicted_label = Model.predict(inp)
        if predicted_label == image_label:
            print(f"Test case {i} passed => Predicted Label: {predicted_label}, Expected Label: {image_label}")
        else: 
            print(f"Test case {i} failed => Predicted Label: {predicted_label}, Expected Label: {image_label}")

    except FileNotFoundError as fe:
        print(f"Test case {i} failed => Invalid Image Path: {image_path}")

    except Exception as e:
        print(f"Test case {i} failed => Unknown Error.")