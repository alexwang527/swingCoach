import cv2 as cv
from pose import PoseEstimator
from numpy.linalg import norm
from numpy import dot
import math
import numpy as np

class DisplayImgManager():

    def __init__(self):
        self.POSE = PoseEstimator()
        self.POSE2 = PoseEstimator()
        self.POSE3 = PoseEstimator()
        self.POSE4 = PoseEstimator()
        self.POSE5 = PoseEstimator()
        self.POSE6 = PoseEstimator()
        self.POSE7 = PoseEstimator()
        self.POSE8 = PoseEstimator()

        self.FIRST = True
    
    def estimate_img(self, am1, pro1, am2, pro2, am3, pro3, am4, pro4):
        img = cv.imread(am1)
        img2 = cv.imread(pro1)
        img3 = cv.imread(am2)
        img4 = cv.imread(pro2)
        img5 = cv.imread(am3)
        img6 = cv.imread(pro3)
        img7 = cv.imread(am4)
        img8 = cv.imread(pro4)

        img = self.POSE.getangles(img)
        img2 = self.POSE2.getangles(img2)
        img3 = self.POSE3.getangles(img3)
        img4 = self.POSE4.getangles(img4)
        img5 = self.POSE5.getangles(img5)
        img6 = self.POSE6.getangles(img6)
        img7 = self.POSE7.getangles(img7)
        img8 = self.POSE8.getangles(img8)

        angles = []
        angles2 = []
        angles3 = []
        angles4 = []
        angles5 = []
        angles6 = []
        angles7 = []
        angles8 = []

        self.compare(self.POSE, self.POSE2, angles, angles2)
        self.compare(self.POSE3, self.POSE4, angles3, angles4)
        self.compare(self.POSE5, self.POSE6, angles5, angles6)
        self.compare(self.POSE7, self.POSE8, angles7, angles8)
        
        v1 = np.array([math.cos(math.radians(angle)) for angle in angles])
        v2 = np.array([math.cos(math.radians(angle)) for angle in angles2])
        v3 = np.array([math.cos(math.radians(angle)) for angle in angles3])
        v4 = np.array([math.cos(math.radians(angle)) for angle in angles4])
        v5 = np.array([math.cos(math.radians(angle)) for angle in angles5])
        v6 = np.array([math.cos(math.radians(angle)) for angle in angles6])
        v7 = np.array([math.cos(math.radians(angle)) for angle in angles7])
        v8 = np.array([math.cos(math.radians(angle)) for angle in angles8])

        # calculate cosine similarity, values close to 1 indicate high similarity
        similarity = dot(v1, v2)/(norm(v1)*norm(v2)) 
        similarity2 = dot(v3, v4)/(norm(v3)*norm(v4)) 
        similarity3 = dot(v5, v6)/(norm(v5)*norm(v6)) 
        similarity4 = dot(v7, v8)/(norm(v7)*norm(v8)) 

        print("Setup score:    ", similarity)
        print("Topswing score: ", similarity2)
        print("Impact score:   ", similarity3)
        print("Finish score:   ", similarity4)

        Hori = np.concatenate((img, img2), axis=0)
        Hori2 = np.concatenate((img3, img4), axis=0)
        Hori3 = np.concatenate((img5, img6), axis =0)
        Hori4 = np.concatenate((img7, img8), axis =0)

        result = np.concatenate((Hori, Hori2, Hori3, Hori4), axis = 1)

        cv.imshow('golfswingstudy', result)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def compare(self, pose1, pose2, angles1, angles2):
        arr = pose1.get_Arr()
        arr2 = pose2.get_Arr()
        lst = [item for sublist in arr.values() for item in sublist] + [value for value in arr.values() if not isinstance(value, list)]
        lst2 = [item for sublist in arr2.values() for item in sublist] + [value for value in arr2.values() if not isinstance(value, list)]
        for i in range (len(lst)):
            if lst[i] != -1 and lst2[i] != -1:
                angles1.append(lst[i])
                angles2.append(lst2[i])