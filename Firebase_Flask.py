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
        authToken = 'eyJhbGciOiJSUzI1NiIsImtpZCI6IjAzZDA3YmJjM2Q3NWM2OTQyNzUxMGY2MTc0ZWIyZj' \
                    'E2NTQ3ZDRhN2QiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuL' \
                    'mdvb2dsZS5jb20vbXktc3RvcmUtN2NlOTEiLCJhdWQiOiJteS1zdG9yZS03Y2U5MSIsImF1' \
                    'dGhfdGltZSI6MTY5ODMxOTI4OCwidXNlcl9pZCI6Inl5cE50blRoMmdNeVNySGRWQ0dBcWVK' \
                    'bTJ3azIiLCJzdWIiOiJ5eXBOdG5UaDJnTXlTckhkVkNHQXFlSm0yd2syIiwiaWF0IjoxNjk4' \
                    'MzE5Mjg4LCJleHAiOjE2OTgzMjI4ODgsImVtYWlsIjoidGVzdDVAdGVzdC5jb20iLCJlbWFp' \
                    'bF92ZXJpZmllZCI6ZmFsc2UsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZW1haWwiOlsid' \
                    'GVzdDVAdGVzdC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJwYXNzd29yZCJ9fQ.gv2uEB_e' \
                    '6JdOYiIJXlL6P2FyMdiGLToXo0d0FTYuMmK-lin4ou5rSFArNps91hKadg0P_ChShGfCtklVJi' \
                    '4MvI9e3K87Fcrq58CN05lapngcJiWqMYzrCLwxVYmDaIUDRBkTKQgqzPo2m8Egspjwc8Nzew_e' \
                    'T3JDqHWaenogfys6VPNHMko_Rc1kCfm9yS_usZBb_QMue02ptgpyl1OhTybOgAmu3FgUQV-7Tew' \
                    'cpzNBlvaansuVppZUpwSRoDatbs52Q0bHICIZDiqhLE2TY1OJjPN_N6_K0yVJbBLuvsL3h4kzpP' \
                    's7o74FxzWgKJdXOwk1pzvsWoeumoHb607nwg'

        storage_url = "https://firebasestorage.googleapis.com/v0/b/my-store-7ce91.appspot.com/o/users%2Fhistory%2F" \
                      f"{user_id}%2F{file_name}?name={file_name}"
        #user_id = request.form.get('user_id')


        if 'image' in request.files:
            image_file = request.files['image']

            # Load and preprocess the image
            img = load_img(image_file)
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

            # Colorize the image
            out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs))
            out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs))

            # Read the image
            #image_data = image_file.read()

            # Create a URL for the image in Firebase Storage
           # storage_location = f'users/history/{user_id}.json?auth={authToken}'

            # Set up the request headers
            headers = {
                'Content-Type': 'image/jpeg',
                'Authorization': authToken
            }

             # Encode the image data in base64
            image_data_json = base64.b64encode(out_img_siggraph17)



            # Make a POST request to upload the image to Firebase Storage
            response = requests.post(f'{storage_url}', headers=headers, data=image_data_json)

           # print(image_data_json)

            if response.status_code == 200:
                #print(print(image_data_json))
                return f'Image uploaded successfully to {storage_url}'
            else:
               # print(image_data_json)
                return f'Failed to upload image. Status code: {response.status_code}'
        else:
            return 'Invalid request. Please provide user_id and the grayscale image in the request.'
    except Exception as e:
        return f'An error occurred: {str(e)}'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
