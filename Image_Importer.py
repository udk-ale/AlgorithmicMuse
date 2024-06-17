from PIL import Image
import numpy as np


def image_to_bw_array(image_path):
    # Open the image file
    img = Image.open(image_path)

    '''
    # Display the original image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    '''

    # Resize the image to 640x640 using LANCZOS for high-quality downsampling
    img = img.resize((640, 640), Image.Resampling.LANCZOS)

    # Convert the image to grayscale
    img = img.convert('L')

    # Convert the grayscale image to binary (black and white)
    threshold = 128  # You can adjust the threshold value based on your needs
    img = img.point(lambda x: 0 if x < threshold else 255)  # Ensure output is 0 or 255 directly

    # Convert the image to a NumPy array
    img_array = np.array(img, dtype=np.uint8)

    # Ensure the array has the correct shape (640, 640, 1)
    img_array = img_array.reshape((640, 640, 1))

    '''
    # Display the transformed image
    plt.subplot(1, 2, 2)
    plt.imshow(img_array[:, :, 0], cmap='gray')
    plt.title('Transformed to 640x640 B&W')
    plt.axis('off')
    plt.show()
    '''
    
    return img_array

# Example usage:
#file_path_image = "TestImage.png"
#image_array = image_to_bw_array(file_path_image)