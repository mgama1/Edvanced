import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import re
def load_models():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten", 
        device_map="auto"  
    )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return processor, model

class BubbleSheetProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.nc = None
        self.row_threshold=10
        self.image = None
        self.gray = None
        self.warped = None
        self.cropped = None
        self.thresh = None
        self.bubble_contours = None
        self.answer_matrix = None
        self.student_id=None
        self.bubbles_area_x, self.bubbles_area_y, self.bubbles_area_w, self.bubbles_area_h=None,None,None,None
        self.marker_area=None

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_id_corner_map(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, _ = detector.detectMarkers(self.gray)
        if ids is None or len(ids) < 4:
            raise ValueError("Not all 4 ArUco markers were detected.")

        id_corner_map = {id[0]: corner for id, corner in zip(ids, corners)}

        if 48 in id_corner_map:
            self.nc = 5
        elif 49 in id_corner_map:
            self.nc = 4
        else:
            raise ValueError("Required ArUco marker not found.")
        #give the area of a circle with diameter equals to the edge of a marker
        marker_arr=id_corner_map[30]
        self.marker_area=np.pi*(int(abs(marker_arr[0][1][0]-marker_arr[0][0][0]))/2)**2
        return id_corner_map

    def get_center(self, corner):
        return corner[0].mean(axis=0)

    def get_centroid(self, array):
        return np.array(array).mean(axis=0)

    def get_roi(self, id_corner_map):
        if 48 in id_corner_map:
            ordered_ids = [30, 10, 48, 34]  # TL, TR, BR, BL
        else:  # 49 is guaranteed if 48 isn't
            ordered_ids = [30, 10, 49, 34]

        ordered_pts = [self.get_center(id_corner_map[id]) for id in ordered_ids]
        return np.array(ordered_pts, dtype='float32')

    def get_warped_image(self, src_pts):
        width, height = int(self.image.shape[0] * 0.764), self.image.shape[0]
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(self.image, M, (width, height))

    def get_largest_contour(self, image, drawContour=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            25, 4
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)

        if drawContour:
            cv2.drawContours(image, largest_contour, -1, (0, 255, 0), 3)
            cv2.imshow('largest contour', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return largest_contour
    
    def warp_paper(self):
        id_corner_map = self.get_id_corner_map()
        roi = self.get_roi(id_corner_map)
        self.warped = self.get_warped_image(roi)

    def preprocess_cropped(self,debug=False):
        self.bubbles_area_x, self.bubbles_area_y, self.bubbles_area_w, self.bubbles_area_h = cv2.boundingRect(self.get_largest_contour(self.warped))
        self.cropped = self.warped[self.bubbles_area_y+10:self.bubbles_area_y+self.bubbles_area_h-10, 
                                    self.bubbles_area_x+10:self.bubbles_area_x+self.bubbles_area_w-10]
        
        cropped_gray = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(cropped_gray, (3, 3), 0)

        self.thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            57, 4
        )
        if debug==True:
            print("self.thresh")
            plt.imshow(self.thresh,cmap='gray')

    def extract_bubble_contours(self, debug=False):
        contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bubble_contours = []
        
        # Filter bubble contours as before
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
        
            if area > self.marker_area and 0.85 < aspect_ratio < 1.15:
                bubble_contours.append(c)
        
        # First sort all bubbles by their y-coordinate (top to bottom)
        bubble_contours.sort(key=lambda b: self.get_centroid(b)[0][1])
        
        # Group bubbles into rows based on y-coordinate proximity
        if not bubble_contours:
            return []  # no bubbles found
        
        rows = []
        current_row = [bubble_contours[0]]
        y_centroid = self.get_centroid(bubble_contours[0])[0][1]
        
        for b in bubble_contours[1:]:
            current_y = self.get_centroid(b)[0][1]
            # If y-coordinate is close enough to current row, add to same row
            if abs(current_y - y_centroid) < self.row_threshold:  # adjust threshold as needed
                current_row.append(b)
            else:
                # Sort current row by x-coordinate and add to rows
                current_row.sort(key=lambda b: self.get_centroid(b)[0][0])
                rows.append(current_row)
                # Start new row
                current_row = [b]
                y_centroid = current_y
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda b: self.get_centroid(b)[0][0])
            rows.append(current_row)
        
        # Flatten the rows back into a single list
        sorted_bubbles = [bubble for row in rows for bubble in row]


        if debug:
            output = self.cropped.copy()
            cv2.drawContours(output, bubble_contours, -1, (0, 255, 0), 2)
            plt.imshow(output)

        self.bubble_contours = sorted_bubbles

    def get_filled_bubbles_matrix(self):
        extracted = []

        for c in self.bubble_contours:
            mask = np.zeros(self.thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            masked = cv2.bitwise_and(self.thresh, self.thresh, mask=mask)

            total = cv2.countNonZero(mask)
            filled = cv2.countNonZero(masked)
            ratio = filled / total

            extracted.append(1 if ratio > 0.5 else 0)

        self.answer_matrix = np.array(extracted).reshape(-1, self.nc)
    def detect_student_id(self,drawContours=False):
        id_area=self.warped[0:self.bubbles_area_y, int(self.warped.shape[1]*.75):]
        x, y, w, h = cv2.boundingRect(self.get_largest_contour(id_area))
        
        student_id_img=id_area[y+5:y+h-5, x+5:x+w-5]
        
        pixel_values = processor(images=student_id_img, return_tensors="pt").pixel_values
        # Run inference
        generated_ids = model.generate(pixel_values,max_length=14,min_length=4,num_beams =2,early_stopping =True,repetition_penalty=.8)
        student_id = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.student_id=re.sub(r'\D', '', student_id)
        if drawContours:
            output = id_area.copy()
            cv2.drawContours(output, self.get_largest_contour(id_area), -1, (0, 255, 0), 2)
            output_resized = cv2.resize(output, (800, int(output.shape[0] * 800 / output.shape[1])))
            cv2.imshow('Detected Bubbles', output_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
    
    def run_pipeline(self):
        self.load_image()
        self.warp_paper()
        self.preprocess_cropped()
        self.extract_bubble_contours()
        self.get_filled_bubbles_matrix()
        self.detect_student_id()
        return self.student_id, self.answer_matrix
