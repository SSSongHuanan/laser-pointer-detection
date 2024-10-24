import cv2
import numpy as np

# Define the distance estimation function
def estimate_distance(area, k):
    # Determine the constant 'k' through calibration
    # Assume distance is inversely proportional to the square root of the area
    if area == 0:
        return None
    distance = k / np.sqrt(area)
    return distance

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Calibration constant
    k = 500  

    while True:
        # Capture each frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Convert the frame to HSV color space for color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the red color range in HSV space
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        # Create masks for the red color range (handling the circular nature of HSV hue)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (9, 9), 0)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If any contours are found, proceed
        if contours:
            # Assume the largest contour is the laser spot
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Filter out small areas to reduce false positives
            if area > 50:
                # Calculate the moments of the contour to get the centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Draw a circle around the detected laser spot
                    cv2.circle(frame, (cX, cY), 15, (0, 255, 0), 2)

                    # Estimate distance
                    distance = estimate_distance(area, k)

                    if distance is not None:
                        # Check if distance is within 2 meters ±10 cm
                        if 1.9 <= distance <= 2.1:
                            distance_text = f"Distance: {distance:.2f} m"
                            cv2.putText(frame, distance_text, (cX - 50, cY - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # Display detection info
                    cv2.putText(frame, "Laser Pointer", (cX - 50, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Laser Pointer Detection', frame)

        # Press 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
