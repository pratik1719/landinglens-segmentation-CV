
LandingLens Segmentation Application Report

Name: Pratik Patil
Project Title: Cat vs Dog Segmentation using LandingLens API (Streamlit Application)

1. Project Introduction
This project shows how to use the LandingLens Segmentation model deployment API to create a semantic segmentation application. By creating a pixel-level mask that highlights the object region, the application aims to segment animals (a dog and a cat) in an input image. When a user uploads a JPG or PNG image, the application uses the Python SDK to send the image to the deployed LandingLens endpoint. The result is a segmentation mask overlay that visually displays the detected object area on the original image. The anticipated result is an overlay mask with a clear color that covers the animal's body area in the picture.

2.
## Sample Input Images
### Dog Sample
![Dog Sample](testing_img/test/5.jpg)
![Cat Sample](testing_img/test/cat/8.jpg)

## Output (Segmentation Result)

![Segmentation Output](testing_img/segmentation_overlay%20(1).png)
![Segmentation Output](testing_img/segmentation_overlay%20(2).png)





4. Demo Video Link
YouTube Demo Video:
🔗( https://www.youtube.com/watch?v=kpnd8IdyeDQ)

5. Application Code Walkthrough
The Streamlit application starts by using a.env file to safely load the API Key and Endpoint ID from environment variables. The Streamlit file uploader is used by the user to upload an image. The uploaded image is sent to the LandingLens deployed endpoint via the LandingAI Python SDK (Predictor) after the "Run Segmentation" button is clicked. Pixel masks from the model's segmentation predictions are decoded and processed to create a colored overlay. The user can then download the result as a PNG file after the app shows the final output image with the segmentation overlay. Additionally, users can test the effects of threshold changes on segmentation accuracy using a confidence/threshold slider.

6. Application Running Result
Using the LandingLens API, the Streamlit application executes live inference with success. The application highlights the detected object region in a segmentation overlay that is returned after an image has been uploaded. The user can test the segmentation output interactively thanks to the interface's threshold adjustment and download features.

7. Short Reflection 

After the API key and endpoint ID were properly configured using environment variables, the API integration went without a hitch. The Streamlit application successfully executed live segmentation inference with minimal problems, and the LandingLens endpoint responded appropriately via the Python SDK.

The segmentation results were obviously influenced by the confidence/threshold setting. The output mask got cleaner when I raised the threshold, but it missed some of the object's boundary regions. Conversely, when I lowered the threshold, more of the object was captured, but occasionally it contained extra noise or the wrong background regions.

If I had more time, I would train the model using a much bigger and more varied dataset in order to enhance its performance. More photos with varying lighting, angles, and backgrounds would improve the model's ability to generalize. I found that the model occasionally overlooked important areas when the dog appeared against a complex background.

