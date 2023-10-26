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



@app.route('/', methods=['GET','POST'])
def colorize():
    try:
        headers = request.headers
        isPreprocess = headers.get('Preprocess')

        if isPreprocess == "pre":

       # sharp image
               image_file = request.files['image']

               # Create a temporary directory to save the uploaded image
               temp_dir = tempfile.mkdtemp()
               temp_image_path = os.path.join(temp_dir, 'uploaded_image.png')
               image_file.save(temp_image_path)

               # Read the image

               img = cv2.imread(temp_image_path,cv2.IMREAD_GRAYSCALE)

               # Apply Gaussian blur to the color image
               blurred = cv2.GaussianBlur(img, (0, 0), 3)

               # Calculate the Unsharp mask by subtracting the blurred image from the original
               unsharp_mask = cv2.addWeighted(img, 1.5, blurred, -0.5, -10)

               # Save the Contrast image as a temporary file
               temp_image_path = os.path.join(temp_dir, 'unsharp_image.png')
               cv2.imwrite(temp_image_path, unsharp_mask)

               # Open the saved Contrast image using the default image viewer on Windows
               os.startfile(temp_image_path)

                #End sharp ##

                # colorize image ##
                # Load and preprocess the image
               img = load_img(temp_image_path)
               (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

                # Colorize the image
               out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs))
               out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs))

                # Save the colorized image to the desktop

               desktop = os.path.expanduser("~/Desktop")
               output_path = os.path.join(desktop, "colorized_image.png")
               plt.imsave(output_path, out_img_siggraph17)

                # Send the colorized image as a response
               return send_file(output_path, mimetype='image/png')

        else :
            # Set the image path
            #image_path = r'C:\Users\gaith\Desktop\colorization\imgs\happy_dog.jpg'

            image_path = request.files['image']
            # Load and preprocess the image
            img = load_img(image_path)
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

            # Colorize the image
            out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs))
            out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs))

            # Save the colorized image to the desktop

            desktop = os.path.expanduser("~/Desktop")
            output_path = os.path.join(desktop, "colorized_image.png")
            plt.imsave(output_path, out_img_siggraph17)

            # Send the colorized image as a response
            return send_file(output_path, mimetype='image/png')
    except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "Failed"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
#
# _________________________________________________________________________________________________






