import base64
import binascii

# Your Base64 string (ensure it is complete and properly formatted)
base64_string = "SGVsbG8gd29ybGQh"

# Log length information
print(f"Length of Base64 string: {len(base64_string)}")
print(f"Length modulo 4: {len(base64_string) % 4}")

# Add padding if necessary
padding_needed = len(base64_string) % 4
if padding_needed == 2:
    base64_string += '=='
elif padding_needed == 3:
    base64_string += '='

# Decode the Base64 string
try:
    decoded_data = base64.b64decode(base64_string)
    
    # Save the decoded data as a .wav file
    with open('output.wav', 'wb') as wav_file:
        wav_file.write(decoded_data)

    print("WAV file has been saved as 'output.wav'.")
except (binascii.Error, ValueError) as e:
    print(f"Error decoding Base64 string: {e}")
