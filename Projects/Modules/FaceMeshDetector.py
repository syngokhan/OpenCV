import cv2
import mediapipe as mp
import numpy as np
import time
import math

class FaceMeshDetector:

	def __init__(self,mode = False,maxFaces = 1,landmarks = False,detectionCon = 0.5,trackCon = 0.5):

		self.mode = mode
		self.maxFaces = maxFaces
		self.landmarks = landmarks
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpDraw = mp.solutions.drawing_utils
		self.mpMesh = mp.solutions.face_mesh

		self.Mesh = self.mpMesh.FaceMesh(self.mode,self.maxFaces,self.landmarks,self.detectionCon,self.trackCon)

		self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1 , circle_radius = 2, color = (0,255,0))

	def findFaceMesh(self,image,draw = True,id_draw = True):

		self.imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		self.results = self.Mesh.process(self.imageRGB)

		self.faces = []
		h,w,c = image.shape

		if self.results.multi_face_landmarks:

			for faceLms in self.results.multi_face_landmarks:
				if draw :
					self.mpDraw.draw_landmarks(image,faceLms,self.mpMesh.FACEMESH_CONTOURS,
						 					   self.drawSpec,self.drawSpec)

				face = []
				for id , lm in enumerate(faceLms.landmark):
					x,y = int(lm.x*w),int(lm.y*h)
					face.append([x,y])

					if id_draw:
						cv2.putText(image,str(id),(x,y+20),cv2.FONT_HERSHEY_PLAIN,.8,(0,0,255),1)

				self.faces.append(face)

		return image,self.faces


	def findDistance(self,p1,p2,image = None):

		x1,y1 = p1
		x2,y2 = p2
		cx,cy = (x1+x2) // 2, (y1+y2) // 2
		length = math.hypot(x2-x1,y2-y1)
		info = (x1,y1,x2,y2,cx,cy)

		if image is not None:

			cv2.circle(image,(x1,y1),5,(255,0,255),cv2.FILLED)
			cv2.circle(image,(x2,y2),5,(255,0,255),cv2.FILLED)
			cv2.circle(image,(cx,cy),5,(255,0,255),cv2.FILLED)

			cv2.line(image,(x1,y1),(x2,y2),(255,0,255),2)
			return length,info,image
		else:
			return length,info


	def findSelectedDraw(self,image,List,color):

		imageBlack = np.zeros((image.shape),np.uint8)
		imageColor = image.copy()

		cropList = []
		for i in List:
			x,y = self.faces[0][i]
			cropList.append([x,y])

		cropList = np.array(cropList)

		imageColor[:] = color

		imageCrop = cv2.fillPoly(imageBlack,[cropList],(255,255,255))

		mask = cv2.bitwise_and(imageColor,imageCrop)

		imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		imageGray = cv2.cvtColor(imageGray,cv2.COLOR_GRAY2BGR)

		results = cv2.addWeighted(imageGray,0.6,mask,0.4,1)
		return results

def main():
	
	cap = cv2.VideoCapture(0)
	cap.set(3,640)
	cap.set(4,480)

	mesh_detector = FaceMeshDetector()

	def empty(a):
		pass

	cv2.namedWindow("BGR")
	cv2.resizeWindow("BGR",1280,240)
	cv2.createTrackbar("Blue","BGR",0,255,empty)
	cv2.createTrackbar("Green","BGR",0,255,empty)
	cv2.createTrackbar("Red","BGR",0,255,empty)

	while True:

		conn,frame = cap.read()
		fps_ = fps()
		cv2.putText(frame,f"FPS : {int(fps_)}",(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

		frame,faces = mesh_detector.findFaceMesh(frame,draw = True,id_draw = False)
		
		if faces:

			face = faces[0]

			length,info,frame = mesh_detector.findDistance(face[63],face[293],frame)

			b = cv2.getTrackbarPos("Blue","BGR")
			g = cv2.getTrackbarPos("Green","BGR")
			r = cv2.getTrackbarPos("Red","BGR")

			results = mesh_detector.findSelectedDraw(frame,[63,293,379,136],color = (b,g,r))

			cv2.imshow("BGR",np.hstack([frame,results]))

		if cv2.waitKey(1) & 0xff == ord("q"):
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
