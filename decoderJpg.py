import base64

def decode_base64_to_jpeg(base64_string, output_file):
    """Decode a Base64 string and save it as a JPEG file.
    
    Args:
        base64_string (str): The Base64 encoded string.
        output_file (str): The path where the JPEG file will be saved.
    """
    try:
        # Remove the header if it exists (data:image/jpeg;base64,)
        if base64_string.startswith("data:image/jpeg;base64,"):
            base64_string = base64_string.split(",")[1]

        # Decode the Base64 string
        image_data = base64.b64decode(base64_string)

        # Write the decoded data to a file
        with open(output_file, 'wb') as file:
            file.write(image_data)

        print(f"Image saved as {output_file}")
    except Exception as e:
        print(f"Error decoding Base64 string: {e}")

# Example usage
base64_image = input("BASE_64 :")  # Replace with your actual Base64 string
output_file_path = "output_image.jpg"
decode_base64_to_jpeg(base64_image, output_file_path)
