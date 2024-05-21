# Image Processing App with Streamlit

This project demonstrates an interactive image processing application built with Streamlit, a Python library for creating web apps. The app allows users to upload images, apply various image processing filters, and view the results in real-time.

**Features:**

* Image Upload: Upload images in JPG, JPEG, or PNG format.
* Default Image: Load a default image for quick experimentation.
* Image Processing Filters:
  * High Pass Filter
  * Median Filter
  * Mean Filter
  * Edge Detection Filters:
  * Roberts Edge Detection
  * Prewitt Edge Detection
  * Sobel Edge Detection
  * Morphological Operations:
  * Erosion
  * Dilation
  * Opening
  * Closing
  * Hough Circle Transform: Detect circles in the image.
  * Thresholding Segmentation: Segment the image based on a user-defined threshold.
  * Side-by-Side Comparison: View the original and filtered images side-by-side.
 # GUI 
 ![](<GUI1.png>)
 
**Requirements:**

* Python 3.x
* Streamlit (pip install streamlit)
* OpenCV (pip install opencv-python)
* Running the App:

**Clone this repository.**
* Open a terminal in the project directory.
* Install the required libraries: pip install -r requirements.txt (if requirements.txt exists)
* Run the app: streamlit run main.py
* 
**How to Use the App:**

* Open http://localhost:8501/ in your web browser.
* Use the file uploader to select an image or click "Default Image".
* Apply the desired filters using the buttons provided.
* The filtered image will be displayed next to the original image.
* 
**Further Enhancements:**

* Add sliders or text boxes to adjust filter parameters.
* Include a dropdown menu for selecting different filters.
* Save the filtered image to the user's device.
**Disclaimer:**

This is a basic example and can be extended to include more advanced image processing techniques.

