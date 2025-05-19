import cv2
import os

input_dir = "data/images/train"
output_dir = "data/images_preprocessed/train"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(input_dir, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Binary Threshold (adjust threshold value if needed)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Save preprocessed image
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, thresh)

print("✅ Preprocessing complete.")
