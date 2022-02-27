import cv2
import numpy as np
import mediapipe as mp
import math
import time


class FaceDetector:


	def __init__(self,minDetection = 0.5, modelSelection = 0):

		self.minDetection = minDetection
		self.modelSelection = modelSelection

		self.mpDraw = mp.solutions.drawing_utils
		self.mpFace = mp.solutions.face_detection

		self.Face = self.mpFace.FaceDetection(self.minDetection,self.modelSelection)


	def findFaces(self,image,draw = True,color = (0,255,0)):

		imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.results = self.Face.process(imageRGB)
		h,w,c = image.shape
		bboxs = []

		if self.results.detections:
			#print(self.results.detections)
			for id,detection in enumerate(self.results.detections):
				
				bboxC = detection.location_data.relative_bounding_box
				bbox = int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
				bboxs.append([id,bbox,detection.score])

				if draw :

					cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2]+bbox[0],bbox[3]+bbox[1]),color,2)
					cv2.putText(image,f"Score : {int(detection.score[0]*100)}%",(bbox[0]-10,bbox[1] - 20),
						        cv2.FONT_HERSHEY_PLAIN,2,color,2)

		return image, bboxs


	def fancyDraw(self,image,bbox,length = 50,thickness = 10,color = (0,255,0)):

		x,y,w,h = bbox
		x1,y1 = x+w,y+h

		# Top Left
		cv2.line(image,(x,y),(x+length,y),color,thickness)
		cv2.line(image,(x,y),(x,y+length),color,thickness)

		# Top Right
		cv2.line(image,(x1,y),(x1-length,y),color,thickness)
		cv2.line(image,(x1,y),(x1,y + length),color,thickness)

		# Bottom Left
		cv2.line(image,(x,y1),(x+length,y1),color,thickness)
		cv2.line(image,(x,y1),(x,y1-length),color,thickness)

		# Bottom Right
		cv2.line(image,(x1,y1),(x1-length,y1),color,thickness)
		cv2.line(image,(x1,y1),(x1,y1-length),color,thickness)	

		return image	

def main():
       
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    
    
    while True:
        
        conn,frame = cap.read()
        frame ,bboxs = face_detector.findFaces(frame,draw = True)
        
        if bboxs:
        	print("Face Detections")

        	info = bboxs[0]
        	id = info[0]
        	bbox =info[1]
        	score = info[2] 
        	frame = face_detector.fancyDraw(frame,bbox)

        
        fps_=fps()
        cv2.putText(frame,f"FPS : {int(fps_)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        cv2.imshow("Face",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":

	pTime = 0
	def fps():
		global pTime
		cTime = time.time()
		fps = 1 / (cTime - pTime)
		pTime = cTime
		return fps

	main()