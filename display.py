import cv2 as cv
from pose import PoseEstimator
from numpy.linalg import norm
from numpy import dot
import math
import numpy as np

class DisplayImgManager():

    def __init__(self):
        self.POSES = [PoseEstimator() for _ in range(8)]

        self.FIRST = True
    
    def estimate_img(self, *args):
        imgs = [cv.imread(arg) for arg in args]
        for i, img in enumerate(imgs):
            imgs[i] = self.POSES[i].getangles(img)

        angles = [[] for _ in range(8)]

        self.compare(self.POSES[0], self.POSES[1], angles[0], angles[1])
        self.compare(self.POSES[2], self.POSES[3], angles[2], angles[3])
        self.compare(self.POSES[4], self.POSES[5], angles[4], angles[5])
        self.compare(self.POSES[6], self.POSES[7], angles[6], angles[7])
        
        vs = []
        for angle_list in angles:
            vs.append(np.array([math.cos(math.radians(angle)) for angle in angle_list]))    

        # calculate cosine similarity, values close to 1 indicate high similarity
        similarity = dot(vs[0], vs[1])/(norm(vs[0])*norm(vs[1])) 
        similarity2 = dot(vs[2], vs[3])/(norm(vs[2])*norm(vs[3])) 
        similarity3 = dot(vs[4], vs[5])/(norm(vs[4])*norm(vs[5])) 
        similarity4 = dot(vs[6], vs[7])/(norm(vs[6])*norm(vs[7])) 

        print("Setup score:    ", similarity)
        print("Topswing score: ", similarity2)
        print("Impact score:   ", similarity3)
        print("Finish score:   ", similarity4)

        Hori = np.concatenate((imgs[0], imgs[1]), axis=0)
        Hori2 = np.concatenate((imgs[2], imgs[3]), axis=0)
        Hori3 = np.concatenate((imgs[4], imgs[5]), axis =0)
        Hori4 = np.concatenate((imgs[6], imgs[7]), axis =0)

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