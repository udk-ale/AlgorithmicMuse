import ultralytics
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Checking CPU
ultralytics.checks()

# Assigning trained classifier
trained_yolo_model = YOLO("best.pt")  # load a custom model

# Testing trained classifier
# Define full paths to one of your images to test the output
image_path = "TestImage.png"

# Load the image and predict using the YOLO model
image = Image.open(image_path)
#predictions = trained_yolo_model(image)
#trained_yolo_model.predict()

#trained_yolo_model.predict(image_path, save=False, imgsz=320, conf=0.5)

# Print the raw predictions to understand their structure
#print(predictions)






# Define the class indices
OVERFIELD_CLASS_INDEX = 4  # Adjust this index as needed
SNIPP_OVERFIELD_CLASS_INDEX = 5  # Adjust this index as needed

def calculate_overfield_prob(cv_model, obs, overfield_class_index, snipp_overfield_class_index):
    # Debug: Print the shape and dtype of the observation array
    #print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")

    # Ensure the observation is in RGB format
    if len(obs.shape) == 2:  # Grayscale image
        #print("Image is grayscale")
        rgb_obs = np.stack((obs,)*3, axis=-1)
    elif len(obs.shape) == 3 and obs.shape[2] == 1:  # Single channel image
        #print("Image is single channel")
        rgb_obs = np.concatenate((obs,)*3, axis=-1)
    elif len(obs.shape) == 3 and obs.shape[2] == 3:  # Already RGB
        #print("Image is already RGB")
        rgb_obs = obs
    elif len(obs.shape) == 3 and obs.shape[2] == 4:  # RGBA image
        #print("Image is RGBA, converting to RGB")
        rgb_obs = obs[:, :, :3]  # Drop the alpha channel
    else:
        raise ValueError(f"Unsupported image format: {obs.shape}")

    #print(f"Converted image shape: {rgb_obs.shape}, dtype: {rgb_obs.dtype}")

    pil_image = Image.fromarray(rgb_obs.astype('uint8'), 'RGB')

    results = cv_model(pil_image, verbose=False)
    overfield_prob = 0.0
    snipp_overfield_prob = 0.0

    for r in results:
        # Check if your specific class indices are in the top 5
        if overfield_class_index in r.probs.top5:
            overfield_index = r.probs.top5.index(overfield_class_index)
            overfield_prob = r.probs.top5conf[overfield_index].item()  # Probability of the specific class
        else:
            overfield_prob = 0.0  # Class not in top 5, probability assumed to be 0

        if snipp_overfield_class_index in r.probs.top5:
            snipp_index = r.probs.top5.index(snipp_overfield_class_index)
            snipp_overfield_prob = r.probs.top5conf[snipp_index].item()  # Probability of the specific class
        else:
            snipp_overfield_prob = 0.0  # Class not in top 5, probability assumed to be 0

    return overfield_prob, snipp_overfield_prob

# Testing the function
test_image_path = "20240614_171150__AI__7.23.png"
image = Image.open(test_image_path)
obs = np.array(image)

# Debug: Print the shape of the test image array
#print(f"Test image shape: {obs.shape}")

# Calculate the probabilities
overfield_prob, snipp_overfield_prob = calculate_overfield_prob(
    trained_yolo_model, obs, OVERFIELD_CLASS_INDEX, SNIPP_OVERFIELD_CLASS_INDEX
)

#print(overfield_prob)
#print(f"Snipp Overfield Probability: {snipp_overfield_prob}")
