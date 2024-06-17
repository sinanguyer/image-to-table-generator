import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import os
import re
import cv2
import numpy as np

def remove_table_lines(img):
    # Convert PIL image to OpenCV format
    img_cv = np.array(img.convert('RGB'))

    # Convert to grayscale and invert the image
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        cv2.drawContours(img_cv, [c], -1, (255, 255, 255), 3)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        cv2.drawContours(img_cv, [c], -1, (255, 255, 255), 3)

    # Convert back to PIL image
    img_no_lines = Image.fromarray(img_cv)
    return img_no_lines

def extract_table_from_image(image_path, zoom_scale=1.5):
    # Configuration for Tesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\sguyer1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    
    # Load and preprocess the image
    img = Image.open(image_path)
    
    # Resize image to zoom in
    original_size = img.size
    new_size = (int(original_size[0] * zoom_scale), int(original_size[1] * zoom_scale))
    img = img.resize(new_size, Image.LANCZOS)  # Using LANCZOS resampling for high-quality enlargement

    # Remove table lines
    img = remove_table_lines(img)

    # Convert to grayscale for Tesseract OCR
    img = img.convert('L')

    # Apply a series of filters to enhance the image
    img = img.filter(ImageFilter.MedianFilter())  # Apply a median filter to remove noise
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Increase the contrast for clearer text

    # Additional image enhancements
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)  # Adjust brightness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2)  # Adjust sharpness
    
    # Perform OCR on the processed image without specifying Tesseract config
    text = pytesseract.image_to_string(img)
    print("OCR Text:")
    print(text)

    # Data collection setup
    data = []
    # Regex to extract headers and numerical values
    number_pattern = re.compile(r'(\d+\.?\d*|\.\d+|\d+)', re.IGNORECASE)  # Improved regex for numbers

    # Process extracted text into structured data line by line
    for line in text.split('\n'):
        print("Processing line:", line)  # Debug output to see each processed line
        if line.strip():  # Avoid processing empty lines
            # Extract numbers using the regex, maintaining text alignment
            found_numbers = number_pattern.findall(line)
            text_part = number_pattern.sub('', line).strip()

            # Combine text with numbers into a row
            row = [text_part] + found_numbers
            data.append(row)

    # Determine the number of columns
    num_cols = max(len(row) - 1 for row in data) if data else 0
    column_names = ['Text'] + [f'Number_{i+1}' for i in range(num_cols)]

    # Create DataFrame
    df = pd.DataFrame(data, columns=column_names).fillna('')
    return df

# Usage example
image_path = r'C:\Users\sguyer1\Documents\translate\translate_image_to_dataframe_imagetotable\extract_image\better_results\image.png'
df = extract_table_from_image(image_path)
print(df)

# Save DataFrame to Excel
excel_path = os.path.splitext(image_path)[0] + '.xlsx'
df.to_excel(excel_path, index=False)
print(f"Data saved to {excel_path}")
