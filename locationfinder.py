import cv2
import numpy as np

class BloodBarLocator:
    def __init__(self):
        self.points = []
        self.is_drawing = False
        self.image = None
        self.clone = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points = [(x, y)]
            self.is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            image_copy = self.clone.copy()
            cv2.rectangle(image_copy, self.points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", image_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.points.append((x, y))
            # Draw final rectangle
            cv2.rectangle(self.image, self.points[0], self.points[1], (0, 255, 0), 2)
            cv2.imshow("Image", self.image)
            
            # Calculate coordinates for the selection
            x1, y1 = min(self.points[0][0], self.points[1][0]), min(self.points[0][1], self.points[1][1])
            x2, y2 = max(self.points[0][0], self.points[1][0]), max(self.points[0][1], self.points[1][1])
            
            # Show coordinates
            print(f"Selected Region Coordinates: ({x1}, {y1}, {x2}, {y2})")
            
            # Get gray values in the selected region
            gray_region = cv2.cvtColor(self.image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            print(f"Average Gray Value in Region: {np.mean(gray_region):.2f}")
            print(f"Min Gray Value in Region: {np.min(gray_region)}")
            print(f"Max Gray Value in Region: {np.max(gray_region)}")

    def locate_blood_bar(self, image_path):
        # Read the image
        self.image = cv2.imread(image_path)
        if self.image is None:
            print("Error: Could not load image")
            return
        
        # Create a window and set mouse callback
        self.clone = self.image.copy()
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        
        print("Instructions:")
        print("1. Click and drag to select the blood bar region")
        print("2. Press 'r' to reset selection")
        print("3. Press 'q' to quit")
        
        while True:
            cv2.imshow("Image", self.image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                self.image = self.clone.copy()
                self.points = []
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    locator = BloodBarLocator()
    locator.locate_blood_bar('bloodfinder.png')