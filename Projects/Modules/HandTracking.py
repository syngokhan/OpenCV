import cv2
import mediapipe as mp
import math
import numpy as np
import time

class HandDetector:

	def __init__(self,mode = False,maxHands = 2 ,complexity = 1, detectionCon = 0.5, trackCon = 0.5):

		self.mode = mode
		self.maxHands = maxHands
		self.complexity = complexity
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpDraw = mp.solutions.drawing_utils
		self.mpHands = mp.solutions.hands

		self.Hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity,self.detectionCon,self.trackCon)

		self.tipIds = [4,8,12,16,20]

	def findHands(self,image,draw = True,flipType = True):

		imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		self.results = self.Hands.process(imageRGB)

		allHands = []
		h,w,c = image.shape

		if self.results.multi_hand_landmarks:
			for handType,handLms in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):

				myHands = {}
				mylmList = []
				xList = []
				yList = []

				for id,lm in enumerate(handLms.landmark):
					px,py,pz = int(lm.x*w),int(lm.y*h),int(lm.z*w)
					mylmList.append([px,py,pz])
					xList.append(px)
					yList.append(py)

				xmin,xmax = min(xList),max(xList)
				ymin,ymax = min(yList),max(yList)

				bboxW,bboxH = xmax - xmin,ymax - ymin
				bbox = xmin,ymin,xmax,ymax

				cx,cy = bbox[0] + (bboxW // 2) , bbox[1] + (bboxH // 2)

				myHands["lmList"] = mylmList
				myHands["bbox"] = bbox
				myHands["center"] = (cx,cy)

				if flipType:

					if handType.classification[0].label == "Right":
						myHands["type"] = "Left"
					else:
						myHands["type"] = "Right"

				else:

					myHands["type"] = handType.classification[0].label

				allHands.append(myHands)


				if draw:
				
					offset = 30
					self.mpDraw.draw_landmarks(image,handLms,self.mpHands.HAND_CONNECTIONS)
					cv2.rectangle(image,(bbox[0] - offset, bbox[1] - offset ),(bbox[2] + offset,bbox[3] + offset),(0,255,0),2)
					cv2.putText(image,myHands["type"],(bbox[0] - offset, bbox[1]-40),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

		if draw:
			return allHands,image
		else:
			return allHands


	def fingersUp(self,image,myHands,flipType = True,draw = False):

		myHandType = myHands["type"]
		mylmList = myHands["lmList"]
		myBoxx = myHands["bbox"]
		fingers = []

		if flipType:

			if myHandType == "Right":
				if mylmList[self.tipIds[0]][0] > mylmList[self.tipIds[0] - 1][0]:
					fingers.append(1)
				else:
					fingers.append(0)

			else:
				if mylmList[self.tipIds[0]][0] < mylmList[self.tipIds[0] - 1][0]:
					fingers.append(1)
				else:
					fingers.append(0)

		else:

			if myHandType == "Right":
				if mylmList[self.tipIds[0]][0] < mylmList[self.tipIds[0] - 1][0]:
					fingers.append(1)
				else:
					fingers.append(0)
			else:

				if mylmList[self.tipIds[0]][0] > mylmList[self.tipIds[0] - 1][0]:
					fingers.append(1)
				else:
					fingers.append(0)

		for id in range(1,5):

			if mylmList[self.tipIds[id]][1] < mylmList[self.tipIds[id] - 2][1]:
				fingers.append(1)
			else:
				fingers.append(0)

		if draw:
			offset = 30	
			cv2.putText(image,f"Fingers : {fingers}",(myBoxx[0] - offset,myBoxx[1] - 70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

		return fingers 

	def findDistance(self,p1,p2,image = None,color = (255,0,255)):

		x1,y1 = p1
		x2,y2 = p2
		cx,cy = (x1 + x2 ) // 2, (y1 + y2) // 2
		length = math.hypot(x2-x1,y2-y1)
		info = (x1,y1,x2,y2,cx,cy)

		if image is not None:

			cv2.circle(image,(x1,y1),15,color,cv2.FILLED)
			cv2.circle(image,(x2,y2),15,color,cv2.FILLED)
			cv2.circle(image,(cx,cy),15,color,cv2.FILLED)
			cv2.line(image,(x1,y1),(x2,y2),color,3)
			return length,info,image
		return length,info


def main():

	cap = cv2.VideoCapture(0)

	hand_detector = HandDetector(maxHands = 2)


	while True:

		conn,frame = cap.read()
		# cv2.flip() kullanmazsanız flipType = False olarak ayarlayın !!!
		frame = cv2.flip(frame,1)

		allHands,frame = hand_detector.findHands(frame,draw = True, flipType = False)

		if allHands:

			hands1 = allHands[0]
			type1 = hands1["type"]
			lmList1 = hands1["lmList"]
			bbox1 = hands1["bbox"]
			center1 = hands1["center"]

			fingers1 = hand_detector.fingersUp(frame,hands1,flipType = False ,draw  = True)

			length1,info1,frame = hand_detector.findDistance(lmList1[4][:-1],lmList1[8][:-1],frame)

			if len(allHands) == 2:

				hands2 = allHands[1]
				type2 = hands2["type"]
				lmList2 = hands2["lmList"]
				bbox2 = hands2["bbox"]
				center2 = hands2["center"]

				fingers2 = hand_detector.fingersUp(frame,hands2,flipType = False ,draw  = True)
				length2,info2,frame = hand_detector.findDistance(lmList2[4][:-1],lmList2[8][:-1],frame)

				# Right and Left Merge !!
				length,info,frame = hand_detector.findDistance(lmList1[4][:-1],lmList2[4][:-1],frame)
				print(length)

		fps_ = fps()
		cv2.putText(frame,f"FPS : {int(fps_)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
		cv2.imshow("Frame",frame)

		if cv2.waitKey(1) & 0xff == ord("q"):
			break

if __name__ == "__main__":

	pTime = 0
	def fps():
		global pTime
		cTime = time.time()
		fps = 1 / (cTime - pTime)
		pTime = cTime
		return fps

	main()