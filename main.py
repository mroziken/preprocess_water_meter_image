import functions_framework
from google.cloud import storage
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
import io
import numpy as np
from scipy.ndimage import label, find_objects

def preprocess_image(image):
    # Convert to grayscale
    gray = image.convert('L')
    
    # Apply Gaussian Blur
    blurred = gray.filter(ImageFilter.GaussianBlur(2))
    
    # Apply thresholding
    thresholded = blurred.point(lambda p: p > 128 and 255)
    
    # Convert image to numpy array
    np_image = np.array(thresholded)
    
    # Find contours using scipy.ndimage
    labeled_array, num_features = label(np_image == 0)
    objects = find_objects(labeled_array)
    
    # Create a mask for the numbers
    mask = Image.new('L', gray.size, 0)
    draw = ImageDraw.Draw(mask)
    
    for obj in objects:
        x1, y1, x2, y2 = obj[1].start, obj[0].start, obj[1].stop, obj[0].stop
        if (x2 - x1) > 10 and (y2 - y1) > 10:  # Filter out small objects
            draw.rectangle([x1, y1, x2, y2], fill=255)
    
    # Bitwise-and to extract the numbers from the original image
    result = Image.composite(gray, Image.new('L', gray.size, 255), mask)
    
    return result

# Register a CloudEvent callback with the Functions Framework that will
# be triggered by Cloud Storage.
@functions_framework.cloud_event
def process_image(cloud_event):
    print(f"Event ID: {cloud_event['id']}")
    print(f"Event Type: {cloud_event['type']}")

    file_data = cloud_event.data
    print(f"Bucket: {file_data['bucket']}")
    print(f"File: {file_data['name']}")
    print(f"Metageneration: {file_data['metageneration']}")
    print(f"Created: {file_data['timeCreated']}")
    print(f"Updated: {file_data['updated']}")

    input_bucket_name = file_data['bucket']
    output_bucket_name = 'garden-watermeter-readings'
    file_name = file_data['name']


    try:
        client = storage.Client()
        input_bucket = client.bucket(input_bucket_name)
        blob = input_bucket.blob(file_name)
        file_buffer = io.BytesIO(blob.download_as_bytes())

        image = Image.open(file_buffer)

        # Image processing steps
        # Rotate the image by 7.5 degrees
        image = image.rotate(7.5, expand=False)
        image = image.crop((530, 670, 880, 740))  # Crop
        image = image.rotate(180, expand=False)
        image = ImageOps.autocontrast(image)
        #Split into black and red
        black_numbers_image = image.crop((0, 0, 200, 70))
        red_numbers_image = image.crop((201, 1, 350, 70))
        
        red_numbers_image = red_numbers_image.convert('RGB')  # Ensure image is in RGB mode
        # Replace red points with black
        pixels = np.array(red_numbers_image)
        red_threshold = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 100) & (pixels[:, :, 2] < 100)
        pixels[red_threshold] = [0, 0, 0]      
        red_numbers_image = Image.fromarray(pixels)
        
        image = Image.new('RGB', (350, 70))
        image.paste(black_numbers_image, (0,0))
        image.paste(red_numbers_image, (200,1))
        
        #image = image.convert('L')  # Greyscale
        
        #threshold = 128
        #image = image.point(lambda p: p > threshold and 255)  # Thresholding

        # Preprocess the merged image for OCR
        image = preprocess_image(image)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        processed_buffer = io.BytesIO()
        image.save(processed_buffer, format='JPEG')
        processed_buffer.seek(0)

        output_bucket = client.bucket(output_bucket_name)
        output_blob = output_bucket.blob(file_name)
        output_blob.upload_from_file(processed_buffer, content_type='image/jpeg')

        print(f"Processed image saved to {output_bucket_name}/{file_name}")
    except Exception as error:
        print('Failed to process image:', error)

