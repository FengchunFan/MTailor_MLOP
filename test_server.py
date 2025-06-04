# Codebase to make a call to the model deployed on the Cerebrium (Note: This should test deployment not something on your local machine)
#   This should accept the path of the image and return/print the id of the class the image belongs to
#   And also accept a flag to run preset custom tests, something like test.py but uses deployed model.
#   Add more tests to test the Cerebrium as a platform. Anything to monitor the deployed model.

import requests
import argparse

# Cerebrium endpoint and API key
URL = "https://api.cortex.cerebrium.ai/v4/p-52b1a8cd/cere-mtailor-mlop/predict"
API_KEY = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTUyYjFhOGNkIiwiaWF0IjoxNzQ4OTkwODMxLCJleHAiOjIwNjQ1NjY4MzF9.2-_evn6DBrkvgCVYlf9yLIuOcB5jbG42iGODvwa8BxcEAs5GqZ5jjcuJngI5S_Z4vq7iAZeUi8C821RY23AwA15dBHZNSB6hCX3p8QPsKphgt0Oai2wP3hXNSapLdVrygQ_W4Nvkp9ZWPjoxrNy-QtBMEVMVmsXZ2KX1WsQmafcgLSvbcUD-xZR2OyYOKZr5rNOFz186foEucA03uyQJNhHnLIleH9QK35Nawr8HhXQPH6s-fRLoWIsu-S4h8j2GjLaEM25Hd0eJRbivy2nXT1acyJQFbG9RJ6wU_XE9SAxphylQpk0wN3JZvydYFF8br_Y4qPxhSwj2u_eD9o3k4Q"
HEADERS = {
    "Authorization": API_KEY
}

# With a flag, load test cases from a testing file
def load_test_cases(filepath):
    namespace = {}
    with open(filepath, "r") as f:
        code = f.read()
        exec(code, {}, namespace)
    return namespace["test_cases"]

# Get result from deployed model
# Input the image_path
def get_cloud_prediction(image_path):
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (image_path, image_file, "image/jpeg")}
            response = requests.post(URL, headers=HEADERS, files=files)
            return response.json()
    except FileNotFoundError:
        return {"error": f"Invalid path: {image_path}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Load image directly
    parser.add_argument("image_path", nargs="?")
    # Load test case from document
    parser.add_argument("-f", "--file", type=str)
    args = parser.parse_args()

    if args.image_path:
        result = get_cloud_prediction(args.image_path)
        predicted_label = result.get("predicted_label")
        print(f"Predicted Label: {predicted_label}")

    elif args.file:
        print(f"Loading custom test cases from {args.file}:")
        print("Test result from local model: ")
        test_cases = load_test_cases(args.file)

        print("Test result from cloud model: ")
        for i, (image_path, image_label) in enumerate(test_cases):
            try:
                result = get_cloud_prediction(image_path)
                # Parse predicted label from result
                # print(result)
                predicted_label = result.get("predicted_label")
                if predicted_label == image_label:
                    print(f"Test case {i} passed => Predicted Label: {predicted_label}, Expected Label: {image_label}")
                else: 
                    print(f"Test case {i} failed => Predicted Label: {predicted_label}, Expected Label: {image_label}")

            except FileNotFoundError as fe:
                print(f"Test case {i} failed => Invalid Image Path: {image_path}")

            except Exception as e:
                print(f"Test case {i} failed => Unknown Error.")

    else:
        print("Please provide either an image path or a test file with -f.")