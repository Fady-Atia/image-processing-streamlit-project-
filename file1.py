import streamlit as st
import cv2
import numpy as np

def load_image(image_path=None, image_bytes=None):
    """Loads an image from either a file path or bytes data."""
    try:
        if image_path is not None:
            image = cv2.imread(image_path)
        elif image_bytes is not None:
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Either 'image_path' or 'image_bytes' must be provided.")

        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def apply_high_pass_filter(image, kernel_size):
    """Applies a high pass filter to the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    hpf_image = cv2.subtract(gray_image, blurred_image)
    return cv2.cvtColor(hpf_image, cv2.COLOR_GRAY2BGR)

def apply_median_filter(image):
    """Applies a median filter to the image."""
    return cv2.medianBlur(image, 9)  # You can adjust the kernel size as needed

def apply_mean_filter(original_image):
    """Applies a mean filter with a kernel size of 5 to the original image."""
    kernel_size = 5
    # Apply mean filter to the original image with the fixed kernel size
    mean_image = cv2.blur(original_image, (kernel_size, kernel_size))
    # Return the mean filtered image
    return mean_image
def apply_roberts_edge_detector(original_image):
    """Applies the Roberts edge detector to detect edges in the original image."""
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Apply Roberts edge detection
    roberts_image = cv2.Canny(gray_image, 100, 200)
    # Return the Roberts edge detected image
    return roberts_image 
def apply_prewitt_edge_detector(original_image):
    """Applies the Prewitt edge detector to detect edges in the original image."""
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Compute the horizontal and vertical gradients using Prewitt operators
    prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    # Compute the magnitude of gradients
    prewitt_image = np.sqrt(prewitt_x**2 + prewitt_y**2)
    # Normalize the gradient magnitude image
    prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Return the Prewitt edge detected image
    return prewitt_image
def apply_sobel_edge_detector(original_image):
    """Applies the Sobel edge detector to detect edges in the original image."""
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Compute the horizontal and vertical gradients using Sobel operators
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    # Compute the magnitude of gradients
    sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalize the gradient magnitude image
    sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Return the Sobel edge detected image
    return sobel_image

def apply_erosion(original_image, kernel_size=5):
    """Applies erosion operation to the image with a specified kernel size."""
    try:
        # Create a kernel for erosion
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform erosion on the original image
        erosion_image = cv2.erode(original_image, kernel, iterations=1)
        return erosion_image
    except Exception as e:
        print(f"Error applying erosion operation: {e}")
        return None    

def apply_dilation(original_image, kernel_size=5):
    """Applies dilation operation to the image with a specified kernel size."""
    try:
        # Create a kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform dilation on the original image
        dilation_image = cv2.dilate(original_image, kernel, iterations=1)
        return dilation_image
    except Exception as e:
        print(f"Error applying dilation operation: {e}")
        return None   
def apply_open(original_image, kernel_size=5):
    """Applies opening operation to the image with a specified kernel size."""
    try:
        # Create a kernel for opening
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform opening on the original image
        open_image = cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel)
        return open_image
    except Exception as e:
        print(f"Error applying opening operation: {e}")
        return None     
def apply_close(original_image, kernel_size=5):
    """Applies closing operation to the image with a specified kernel size."""
    try:
        # Create a kernel for closing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Perform closing on the original image
        close_image = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel)
        return close_image
    except Exception as e:
        print(f"Error applying closing operation: {e}")
        return None
def apply_hough_circle_transform(original_image):
    """Applies Hough circle transform to detect circles in the image."""
    try:
        # Convert the original image to grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # Detect circles using Hough circle transform with specified parameters
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        # Check if any circles are detected
        if circles is not None:
            # Convert the circle parameters to integer
            circles = np.uint16(np.around(circles))
            # Create a copy of the original image for drawing circles
            hough_image = original_image.copy()
            # Draw detected circles on the image
            for i in circles[0, :]:
                cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the outer circle
                cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)       # Draw the center of the circle
            return hough_image
        else:
            return None
    except Exception as e:
        print(f"Error applying Hough circle transform: {e}")

        return None
    
def apply_thresholding_segmentation(original_image):
  """Applies thresholding segmentation to an image.

  Args:
      original_image (numpy.ndarray): The original image in BGR format.

  Returns:
      numpy.ndarray: The thresholded image in BGR format.
  """

  gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
  threshold_value = st.slider("Threshold", 0, 255, 127)  # Get threshold from slider
  _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
  return cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)    
   

def main():
    st.title('Image Processing Project')
    """Displays the image based on button clicks."""
    default_image = load_image('apple.jpg')

    # Initialize session state for current_image and filtered_image
    if 'current_image' not in st.session_state:
        st.session_state['current_image'] = default_image
    if 'filtered_image' not in st.session_state:
        st.session_state['filtered_image'] = 0 #apply_high_pass_filter(default_image, 9)  # Default kernel size 

    # Show a temporary message while image is loading (optional)
    loading_text = st.empty()

    # File uploader for image
    uploaded_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    default_button = st.button('Default Image')
    update_button = st.button('Update Image')
    
    #display buttons 
    col1, col2, col3,col4 = st.columns(4)
    with col1:
    
        high_pass_button = st.button('Apply HP Filter')
        median_filter_button = st.button('Apply Median Filter')
        mean_filter_button=st.button('Apply Mean Filter')
    with col2:
        robert_button=st.button('Apply Robert Filter')
        prewitt_button=st.button('Apply Prewitt Filter')
        sobel_button=st.button('Apply Sobel Filter')
    with col3:    
        erosion_button=st.button('Apply Erosion Filter')
        dilation_button=st.button('Dialation Filter')
        open_button=st.button('Apply Open')
    with col4:   
       close_button=st.button('Apply close Filter')
       hough_circle_button=st.button('Apply Houph circle')
       threeshold_segmentation_button=st.button('Threeshold Segmentation')





    # Display default image when "Default Image" button is clicked
    if default_button:
        st.session_state['current_image'] = default_image
        st.session_state['filtered_image'] = default_image

    # Update image when "Update Image" button is clicked and a file is uploaded
    if update_button and uploaded_image is not None:
        try:
            # Read the uploaded image as bytes
            image_bytes = uploaded_image.read()

            # Show loading text (optional)
            loading_text.text("Loading image...")

            # Load the uploaded image
            updated_image = load_image(image_bytes=image_bytes)

            # Update the state variable with the uploaded image
            st.session_state['current_image'] = updated_image
            st.session_state['filtered_image'] = updated_image 

            # Clear loading text (optional)
            loading_text.empty()
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")

    # Apply high pass filter to the current image with the selected kernel size
    if high_pass_button:
        st.session_state['filtered_image'] = apply_high_pass_filter(st.session_state['current_image'], 9)

    # Apply median filter to the current image
    if median_filter_button:
        st.session_state['filtered_image'] = apply_median_filter(st.session_state['current_image'])

    if mean_filter_button:
        st.session_state['filtered_image'] = apply_mean_filter(st.session_state['current_image']) 

    if robert_button:
        st.session_state['filtered_image'] = apply_roberts_edge_detector(st.session_state['current_image']) 
    if prewitt_button:
        st.session_state['filtered_image']=apply_prewitt_edge_detector(st.session_state['current_image'])
    if sobel_button:
        st.session_state['filtered_image']=apply_sobel_edge_detector(st.session_state['current_image'])

    if erosion_button:
        st.session_state['filtered_image']=apply_erosion(st.session_state['current_image'])  
    if dilation_button :
        st.session_state['filtered_image']=apply_dilation(st.session_state['current_image'])

    if open_button:
        st.session_state['filtered_image']=apply_open(st.session_state['current_image'])
    if close_button:
        st.session_state['filtered_image']=apply_close(st.session_state['current_image'])   
    if hough_circle_button:
        st.session_state['filtered_image']=apply_hough_circle_transform(st.session_state['current_image'])   
    if  threeshold_segmentation_button:
            st.session_state['filtered_image']=apply_thresholding_segmentation(st.session_state['current_image'])   







    # Display the current image and the filtered image side by side
    if st.session_state.get('current_image') is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(st.session_state['current_image'], channels='BGR' ,caption='Original Image', width=300)
            
    

        with col2:
            st.image(st.session_state['filtered_image'], caption='Filtered Image', width=300)
            

if __name__ == "__main__":
    main()
