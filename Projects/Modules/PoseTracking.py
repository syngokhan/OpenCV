import cv2
import mediapipe as mp
import numpy as np
import math
import time


class PoseDetector:

	def __init__(self,mode = False,complexity = 1, smooth = True,enable_seg = False,smooth_seg = True,
				detectionCon = 0.5,trackCon = 0.5):

		self.mode = mode
		self.complexity = complexity
		self.smooth = smooth
		self.enable_seg = enable_seg
		self.smooth_seg = smooth_seg
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpDraw = mp.solutions.drawing_utils
		self.mpPose = mp.solutions.pose

		self.Pose = self.mpPose.Pose(self.mode,self.complexity,self.smooth,self.enable_seg,
									  self.smooth_seg,self.detectionCon,self.trackCon)

	def findPose(self,image,draw = True):

		imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		self.results = self.Pose.process(imageRGB)

		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(image,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

		return image

	def findPosition(self,image,draw = True,bboxWithHands = False):

		self.lmList = []
		self.bboxInfo = {}
		h,w,c = image.shape

		if self.results.pose_landmarks:
			for id,lm in enumerate(self.results.pose_landmarks.landmark):
				cx,cy,cz = int(lm.x*w),int(lm.y*h),int(lm.z*w)
				self.lmList.append([id,cx,cy])

			ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2

			if bboxWithHands:

				x1 = self.lmList[16][1] - ad
				x2 = self.lmList[15][1] + ad
			else:

				x1 = self.lmList[12][1] - ad
				x2 = self.lmList[11][1] + ad

			y2 = self.lmList[29][2] + ad
			y1 = self.lmList[1][2] - ad

			bbox = (x1,y1,x2,y2)

			cx,cy = self.lmList[12][1] + ad, abs(self.lmList[29][2] - self.lmList[1][2]) // 2

			self.bboxInfo = {"bbox" : bbox, "center" : (cx,cy)}

			if draw:
				cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),3)
				cv2.circle(image,(cx,cy),5,(255,0,0),cv2.FILLED)

		return self.lmList,self.bboxInfo

	def findAngle(self,image,p1,p2,p3,draw = True):

		x1,y1 = self.lmList[p1][1:]
		x2,y2 = self.lmList[p2][1:]
		x3,y3 = self.lmList[p3][1:]


		angle = math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2)
		angle = math.degrees(angle)

		if angle < 0:
			angle+=360

		 # Draw
		if draw:
			cv2.putText(image,f"Angle : {int(angle)}",(x2-100,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
			
			cv2.line(image,(x1,y1),(x2,y2),(255,255,255),3)
			cv2.line(image,(x3,y3),(x2,y2),(255,255,255),3)
			
			cv2.circle(image, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
			cv2.circle(image, (x1, y1), 15, (0, 0, 255), 2)
			
			cv2.circle(image, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
			cv2.circle(image, (x2, y2), 15, (0, 0, 255), 2)
			
			cv2.circle(image, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
			cv2.circle(image, (x3, y3), 15, (0, 0, 255), 2)
		
		return angle

	def findDistance(self,image,p1,p2,draw = True,r = 15, t = 3):

		x1,y1 = self.lmList[p1][1:]
		x2,y2 = self.lmList[p2][1:]
		cx,cy = (x1+x2)//2 ,(y1+y2)//2

		if draw:

			cv2.line(image,(x1,y1),(x2,y2),(255,0,255),t)
			cv2.circle(image,(x1,y1),r,(255,0,255),cv2.FILLED)
			cv2.circle(image,(x2,y2),r,(255,0,255),cv2.FILLED)
			cv2.circle(image,(cx,cy),r,(255,0,255),cv2.FILLED)

		length = math.hypot(x2-x1,y2-y1)
		return length,image,[x1,y1,x2,y2,cx,cy]

	def angleCheck(self,myAngle,targetAngle,addOn = 20):
		return targetAngle - addOn < myAngle < targetAngle + addOn


def main():

	path = "/Users/gokhanersoz/Desktop/Hepsi/OpenCV/PROJECTS/Resources/pose4.mp4"
	#cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture(path)
	
	pose_detector = PoseDetector()
	
	while True:
		
		conn,frame = cap.read()

		frame = pose_detector.findPose(frame,draw = False)
		lmList,info = pose_detector.findPosition(frame,draw = True,bboxWithHands = True)
		if lmList:
			print(info)
			angle = pose_detector.findAngle(frame,15,13,11)
			length,frame,_ = pose_detector.findDistance(frame,12,24,draw = True)
			print(length)
			print(angle,pose_detector.angleCheck(angle,100,addOn = 20))

		fps_= fps()
		cv2.putText(frame,f"FPS : {int(fps_)}",(20,70),cv2.FONT_HERSHEY_PLAIN,4,(255,0,0),4)
		cv2.imshow("Frame",frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
		
	cv2.destroyAllWindows()
	cap.release()
	
if __name__ == "__main__":

	pTime = 0
	def fps():
		global pTime
		cTime = time.time()
		fps = 1 / (cTime - pTime)
		pTime = cTime
		return fps
  
	main()