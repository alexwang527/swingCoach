import cv2 as cv
import numpy as np
import math

class PoseEstimator():

    def __init__(self):

        self.BODY_PARTS =  { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, 
               "Background": 14 }
            
        self.POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

        modelfile = "models/pose_iter_160000.caffemodel"
        configfile = "models/pose_deploy_linevec_faster_4_stages.prototxt"
        self.NET = cv.dnn.readNetFromCaffe(configfile, modelfile)

        # for mobileNet
        #self.NET = cv.dnn.readNetFromTensorflow("models/graph_opt.pb")

        self.THR = 0.2
        self.IN_WIDTH = 396
        self.IN_HEIGHT = 368

        self.POINTS = []

        self.KEY_DISTANCES = {"RArm":{"RShoulder-RElbow":None,"RElbow-RWrist":None,"Neck-RShoulder":None},
        "LArm":{"LShoulder-LElbow":None,"LElbow-LWrist":None,"Neck-LShoulder":None},
        "RLeg":{"RHip-RKnee":None,"RKnee-RAnkle":None},
        "LLeg":{"LHip-RKnee":None,"LKnee-RAnkle":None}}

        self.KEY_ANGLES = {"RArm": [],"LArm":[],"RLeg":[],"LLeg":[]}

        self.TEXT_COLOR = (0,0,0)

    def radiansTodeg(self,rad):
        return rad * (180/math.pi)

    def getangles(self, frame, wantBlank = False):
        """applies pose estimation on frame, gets the distances between points"""

        RShoulder_pos = None
        RWrist_pos = None
        LShoulder_pos = None
        LWrist_pos = None
        Neck_pos = None
        RElbow_pos = None
        LElbow_pos = None
        RHip_pos = None
        RAnkle_pos = None
        LHip_pos = None
        LAnkle_pos = None

        frame_h,frame_w = frame.shape[0:2]
            
        self.NET.setInput(cv.dnn.blobFromImage(frame, 1.0 / 255, (self.IN_WIDTH, self.IN_HEIGHT),(0, 0, 0), swapRB=False, crop=False))
        out = self.NET.forward()

        out = out[:, :15, :, :] 

        assert(len(self.BODY_PARTS) == out.shape[1])

        self.POINTS.clear()

        for i in range(len(self.BODY_PARTS)):
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frame_w * point[0]) / out.shape[3]
            y = (frame_h * point[1]) / out.shape[2]

            if(conf > self.THR):
                self.POINTS.append((int(x),int(y)))
            else:
                self.POINTS.append(None)

        if wantBlank:

            frame = np.zeros((frame_h,frame_w,3),np.uint8)

            self.TEXT_COLOR = (255,255,255)

        for pair in self.POSE_PAIRS:

            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in self.BODY_PARTS)
            assert(partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if self.POINTS[idFrom] and self.POINTS[idTo]:

                if(partFrom == "RShoulder"):
                    RShoulder_pos = self.POINTS[idFrom]

                if(partTo == "RWrist"):
                    RWrist_pos = self.POINTS[idTo]

                if(partFrom == "LShoulder"):
                    LShoulder_pos = self.POINTS[idFrom]

                if(partTo == "LWrist"):
                    LWrist_pos = self.POINTS[idTo]

                if(partFrom == "Neck"):
                    Neck_pos = self.POINTS[idFrom]
                
                if(partTo == "RElbow"):
                    RElbow_pos = self.POINTS[idTo]

                if(partFrom == "RElbow"):
                    RElbow_pos = self.POINTS[idFrom]
                
                if(partFrom == "LElbow"):
                    LElbow_pos = self.POINTS[idFrom]

                if(partTo == "LElbow"):
                    LElbow_pos = self.POINTS[idTo]

                if(partFrom == "RHip"):
                    RHip_pos = self.POINTS[idFrom]
                
                if(partTo == "RAnkle"):
                    RAnkle_pos = self.POINTS[idTo]
                    
                if(partFrom == "LHip"):
                    LHip_pos = self.POINTS[idFrom]
                
                if(partTo == "LAnkle"):
                    LAnkle_pos = self.POINTS[idTo]

                if(partFrom == "RShoulder" and partTo == "RElbow"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)
                    
                elif(partFrom == "RElbow" and partTo == "RWrist"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "LShoulder" and partTo == "LElbow"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "LElbow" and partTo == "LWrist"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "Neck" and partTo == "RShoulder"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "Neck" and partTo == "LShoulder"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "RHip" and partTo == "RKnee"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "RKnee" and partTo == "RAnkle"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "LHip" and partTo == "LKnee"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)

                elif(partFrom == "LKnee" and partTo == "LAnkle"):
                    self.distance(idFrom, idTo, partFrom, partTo, self.KEY_DISTANCES)


                cv.line(frame, self.POINTS[idFrom], self.POINTS[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, self.POINTS[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, self.POINTS[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                

        if(RShoulder_pos is not None and RWrist_pos is not None and 
           self.KEY_DISTANCES["RArm"]["RShoulder-RElbow"] is not None and 
           self.KEY_DISTANCES["RArm"]["RElbow-RWrist"] is not None):

            self.calcAngle(RShoulder_pos, RWrist_pos, "RArm", "RShoulder-RElbow", "RElbow-RWrist", self.KEY_ANGLES)
                
        if(LShoulder_pos is not None and LWrist_pos is not None and 
           self.KEY_DISTANCES["LArm"]["LShoulder-LElbow"] is not None and 
           self.KEY_DISTANCES["LArm"]["LElbow-LWrist"] is not None):

            self.calcAngle(LShoulder_pos, LWrist_pos, "LArm", "LShoulder-LElbow", "LElbow-LWrist", self.KEY_ANGLES)

        if(Neck_pos is not None and LElbow_pos is not None and 
           self.KEY_DISTANCES["LArm"]["Neck-LShoulder"] is not None and
           self.KEY_DISTANCES["LArm"]["LShoulder-LElbow"] is not None):

            self.calcAngle(Neck_pos, LElbow_pos, "LArm", "Neck-LShoulder", "LShoulder-LElbow", self.KEY_ANGLES)

        if(Neck_pos is not None and RElbow_pos is not None and 
           self.KEY_DISTANCES["RArm"]["Neck-RShoulder"] is not None and
           self.KEY_DISTANCES["RArm"]["RShoulder-RElbow"] is not None):

            self.calcAngle(Neck_pos, RElbow_pos, "RArm", "Neck-RShoulder", "RShoulder-RElbow", self.KEY_ANGLES)

        if(RHip_pos is not None and RAnkle_pos is not None and 
           self.KEY_DISTANCES["RLeg"]["RHip-RKnee"] is not None and
           self.KEY_DISTANCES["RLeg"]["RKnee-RAnkle"] is not None):

            self.calcAngle(RHip_pos, RAnkle_pos, "RLeg", "RHip-RKnee", "RKnee-RAnkle", self.KEY_ANGLES)

        if(LHip_pos is not None and LAnkle_pos is not None and 
           self.KEY_DISTANCES["LLeg"]["LHip-LKnee"] is not None and 
           self.KEY_DISTANCES["LLeg"]["LKnee-LAnkle"] is not None):

            self.calcAngle(LHip_pos, LAnkle_pos, "LLeg", "LHip-LKnee", "LKnee-LAnkle", self.KEY_ANGLES)

        t, _ = self.NET.getPerfProfile()
        freq = cv.getTickFrequency() / 1000

        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR)
        cv.putText(frame, "Caffe", (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR)

        if len(self.KEY_ANGLES["RArm"]) == 1:
            self.KEY_ANGLES["RArm"].append(-1)
        if len(self.KEY_ANGLES["RArm"]) == 0:
            self.KEY_ANGLES["RArm"].append(-1)
            self.KEY_ANGLES["RArm"].append(-1)

        if len(self.KEY_ANGLES["LArm"]) == 1:
            self.KEY_ANGLES["LArm"].append(-1)
        if len(self.KEY_ANGLES["LArm"]) == 0:
            self.KEY_ANGLES["LArm"].append(-1)
            self.KEY_ANGLES["LArm"].append(-1)
        
        if len(self.KEY_ANGLES["RLeg"]) == 0:
            self.KEY_ANGLES["RLeg"].append(-1)
        
        if len(self.KEY_ANGLES["LLeg"]) == 0:
            self.KEY_ANGLES["LLeg"].append(-1)

        return frame

    def get_Arr(self):
        return self.KEY_ANGLES

    def distance(self, idFrom, idTo, partFrom, partTo, keyDistances):
        dist = math.pow((self.POINTS[idFrom][0] - self.POINTS[idTo][0]), 2) + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
        ft = partFrom + "-" + partTo
        key = ""
        if(partTo == "RElbow" or partTo == "RShoulder" or partTo == "RWrist"):
            key = "RArm"
        elif(partTo == "LElbow" or partTo == "LShoulder" or partTo == "LWrist"):
            key = "LArm"
        elif(partTo == "RKnee" or partTo == "RAnkle"):
            key = "RLeg"
        elif(partTo == "LKnee" or partTo == "LAnkle"):
            key = "LLeg"
        keyDistances[key][ft] = dist

    def calcAngle(self, pos1, pos2, key, val1, val2, keyAngles):
        c_2 = math.pow((pos1[0] - pos2[0]),2) + math.pow((pos1[1] - pos2[1]),2)
        a_2 = self.KEY_DISTANCES[key][val1]
        b_2 = self.KEY_DISTANCES[key][val2]
        theta = self.radiansTodeg(math.acos((a_2 + b_2 - c_2)/(2*math.sqrt(a_2*b_2))))
        keyAngles[key].append(theta)