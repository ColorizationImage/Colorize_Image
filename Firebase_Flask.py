import base64

from flask import Flask, send_file,request
from colorizers import *
import requests

import os
import matplotlib.pyplot as plt
from flask import Flask, send_file,request
from colorizers import *
from demo_release import load_img, preprocess_img, postprocess_tens
import cv2
import tempfile


app = Flask(__name__)

# Load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()


@app.route('/', methods=['POST'])
def colorization():
    try:
        file_name ='test1.png'
        user_id = "FyypNtnTh2gMySrHdVCGAqeJm2wk2"
        authToken = 'eyJhbGciOiJSUzI1NiIsImtpZCI6IjBkMGU4NmJkNjQ3NDBjYWQyNDc1NjI4ZGEyZ' \
                    'WM0OTZkZjUyYWRiNWQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3Vy' \
                    'ZXRva2VuLmdvb2dsZS5jb20vbXktc3RvcmUtN2NlOTEiLCJhdWQiOiJteS1zdG9yZS0' \
                    '3Y2U5MSIsImF1dGhfdGltZSI6MTY5ODMzNzc0OCwidXNlcl9pZCI6Inl5cE50blRoMmdN' \
                    'eVNySGRWQ0dBcWVKbTJ3azIiLCJzdWIiOiJ5eXBOdG5UaDJnTXlTckhkVkNHQXFlSm0yd2' \
                    'syIiwiaWF0IjoxNjk4MzM3NzQ4LCJleHAiOjE2OTgzNDEzNDgsImVtYWlsIjoidGVzdDVAd' \
                    'GVzdC5jb20iLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImZpcmViYXNlIjp7ImlkZW50aXRpZ' \
                    'XMiOnsiZW1haWwiOlsidGVzdDVAdGVzdC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJwY' \
                    'XNzd29yZCJ9fQ.AR8IduM99qmzeceF4PPPfeUv_NcOY11aDw5AQ6p-bcgDoHawGdLdmRA4xz' \
                    'mFJ7Zp8WYwejw3uida2nWv0c7W2YOkGy9W5TQHtlQZUx8BfmujfgMSVglW5El1gu9Vt3UtxH' \
                    'o3z6VHr-No4fwBV0AIucU5VPcmbweyUcWJJXHsq_uBeE2KaPqh2YXWSdT9pza_kfDyyQmzFBL' \
                    'Q-Q2jxATBrC5fVnBDBXSschEBdhuQVo4MtK1GHcVo3zIavVyvjG-QJ7mtlQFuSNnMZbNvg-Vo' \
                    'gh7KxmgUONCacIFRDskAVr1VyuUII_Xr-2u5MRT6YGEh1cjM0e6LUbysIdJQKYMsQA'

        storage_url = "https://firebasestorage.googleapis.com/v0/b/my-store-7ce91.appspot.com/o/users%2Fhistory%2F" \
                      f"{user_id}%2F{file_name}?name={file_name}"
        #user_id = request.form.get('user_id')


        if 'image' in request.files:
            image_file = request.files['image']
            headers = request.headers
            isPreprocess = headers.get('Preprocess')

            if isPreprocess == "pre":
                #sharp image
                # Create a temporary directory to save the uploaded image
                temp_dir = tempfile.mkdtemp()
                temp_image_path = os.path.join(temp_dir, 'uploaded_image.png')
                image_file.save(temp_image_path)

                # Read the image

                img = cv2.imread(temp_image_path, cv2.IMREAD_GRAYSCALE)

                # Apply Gaussian blur to the color image
                blurred = cv2.GaussianBlur(img, (0, 0), 3)

                # Calculate the Unsharp mask by subtracting the blurred image from the original
                unsharp_mask = cv2.addWeighted(img, 1.5, blurred, -0.5, -10)

                # Save the Contrast image as a temporary file
                temp_image_path = os.path.join(temp_dir, 'unsharp_image.png')
                cv2.imwrite(temp_image_path, unsharp_mask)

                # End sharp Image ##

            # Load and preprocess the image
            img = load_img(temp_image_path)
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

            # Colorize the image
            out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs))
            out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs))

            # Create a URL for the image in Firebase Storage
           # storage_location = f'users/history/{user_id}.json?auth={authToken}'


            # Set up the request headers
            headers = {
                'Content-Type': 'image/jpeg',
                'Authorization': authToken
            }

            # Save the colorized image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
                plt.imsave(temp_image, out_img_siggraph17)
                image_data = open(temp_image.name, 'rb').read()



             # Encode the image data in base64
            image_data_json = base64.b64encode(image_data)



            # Make a POST request to upload the image to Firebase Storage
            response = requests.post(f'{storage_url}', headers=headers, data=image_data_json)


           # print(image_data_json)
            if response.status_code == 200:
                #print(print(image_data_json))
                print(f'Image uploaded successfully to {storage_url}')
                return f'Image uploaded successfully to {storage_url}'
            else:
               # print(image_data_json)
                print(f'Failed to upload image. Status code: {response.status_code}')
                return f'Failed to upload image. Status code: {response.status_code}'
        else:
            print('Invalid request. Please provide user_id and the grayscale image in the request.')
            return 'Invalid request. Please provide user_id and the grayscale image in the request.'
    except Exception as e:
        return f'An error occurred: {str(e)}'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
