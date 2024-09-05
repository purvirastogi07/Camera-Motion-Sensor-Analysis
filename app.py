import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify another index for multiple cameras

# Read the first frame
ret, frame1 = cap.read()

# Convert the frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve detection accuracy
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Compute the absolute difference between the two frames
    delta_frame = cv2.absdiff(gray1, gray2)

    # Threshold the delta image to get binary image
    thresh = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the threshold image to fill in holes and find contours
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the binary image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours and draw a rectangle around large motions
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue  # Ignore small movements
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Motion Detector", frame2)

    # Update the first frame for the next iteration
    gray1 = gray2.copy()

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
