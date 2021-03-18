#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from multiprocessing import Process
from multiprocessing import Value

import time

from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

from ardrone_autonomy.srv import CamSelect
import std_srvs.srv

class MoveDrone:	
	def __init__(self):
		self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10) 
		self.landing_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=10)
		self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

	def move_drone(self, speed=[0.0, 0.0, 0.0], orient=[0.0, 0.0, 0.0]):
		vel_msg = Twist()

		vel_msg.linear.x = speed[0]
		vel_msg.linear.y = speed[1]
		vel_msg.linear.z = speed[2]

		vel_msg.angular.x = orient[0]
		vel_msg.angular.y = orient[1]
		vel_msg.angular.z = orient[2]
		self.move_pub.publish(vel_msg)
		return 0

	def takeoff_drone(self):
		empty_msg = Empty()
		
		#print('i can fly')
		self.takeoff_pub.publish(empty_msg)

	def land_drone(self):
		empty_msg = Empty()
		
		self.landing_pub.publish(empty_msg)

	def toggleCam(self):
		rospy.wait_for_service( 'ardrone/togglecam' )
		try:
		    toggle = rospy.ServiceProxy( 'ardrone/togglecam', std_srvs.srv.Empty )
		    toggle()
		except rospy.ServiceException, e:
		    print("Service call failed: %s",e)


class image_converter:

  list_pixel = []

  flag = True

  def __init__(self,link):

    #self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()

    self.image_bottom = rospy.Subscriber("/ardrone/bottom/image_raw",Image,self.callback)

    self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",Image,self.callback)

    self.face_classifier = cv2.CascadeClassifier('/home/corentin/drone_workspace/src/my_package/scripts/haarcascade_frontalface_default.xml')

    self.lien = link


  def callback(self,data):
    cv_image = None
    

    if self.lien.value == 0:
	
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        
        #print("im :\n",cv_image,'\n')
        #print("shape :\n",cv_image.shape,'\n')
    
        
    	cv_image = cv2.resize(cv_image,(400,200))
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.1, 4)
    
        #if faces is ():
           #print("No Face Found")
        for (x,y,w,h) in faces:
           cv2.rectangle(cv_image,(x,y),(x+w,y+h),(127,0,255),2)
           print("width : {} height : {}".format(w,h))
           if w > 78:
             print("BIPBIP")
             self.lien.value = 1
           else:
             self.lien.value = 0
	cv2.imshow("Image window", cv_image)
    	cv2.waitKey(1)
             
    #elif self.lien.value == 2:

        

	#time.sleep(5)

	#try:
          #cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        #except CvBridgeError as e:
          #print(e)
	#print("data : ",cv_image.shape)
        """if len(list_pixel) >= 5:
            list_pixel = list_pixel[1:]
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
          
	#print("hi")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
        pixel = []
        relative_value_i = int(thresh.shape[0]/10)
        relative_value_j = int(thresh.shape[1]/10)
    
        mid_value_i = int(thresh.shape[0]/2)
        mid_value_j = int(thresh.shape[1]/2)
    
        for i in range(mid_value_i - relative_value_i ,mid_value_i + relative_value_i):
            for j in range(mid_value_j - relative_value_j ,mid_value_j + relative_value_j):
                pixel.append(thresh.item(i, j))
                
        list_pixel.append(pixel)
        if len(list_pixel) >= 5:
            if(0 in list_pixel[0] and 0 in list_pixel[1] and 0 in list_pixel[2] and 0 in list_pixel[3] and 0 in list_pixel[4] and flag == True):
                flag = False
    
            if(0 not in list_pixel[0] and 0 not in list_pixel[1] and 0 not in list_pixel[2] and 0 not in list_pixel[3] and 0 not in list_pixel[4] and flag == False):
                flag = True
                self.link.value = 1"""
    #print(cv_image.shape)
    	#cv2.imshow("Image window", cv_image)
    	#cv2.waitKey(1)
    
    #else:
	#print("[{}]".format(self.lien.value),end="")
    #if self.lien.value == 1:
	#print("node 1 : waiting...")


def p1(buffer1):
	ic = image_converter(buffer1)
  	rospy.init_node('image_converter', anonymous=True)
  	try:
    		rospy.spin()
  	except KeyboardInterrupt:
    		print("Shutting down node 1")
  	cv2.destroyAllWindows()

def p2(buffer2):
	rospy.init_node('basic_controller', anonymous=True)
	move = MoveDrone()
	#move.toggleCam()
	
	time_var = time.time()
	while(time.time() < time_var+2):
		move.move_drone(speed=[0.0,0.0,0.0])
		pass

	print("begin")
	time_var = time.time()
	while(time.time() < time_var+3):
		move.takeoff_drone()
		pass
	
	move.move_drone(speed=[0.0,0.0,0.0])

	time_var = time.time()
	while(time.time() < time_var + 3):
		move.move_drone(speed=[0.0,0.0,0.5])
		pass

	move.move_drone(speed=[0.0,0.0,0.0])

	#routine
	time_var = time.time()
	#last_state = None
	TIME_OF_RUNNING = 40 #seconds

	while (time.time() < time_var+TIME_OF_RUNNING):
		#print(last_state, buffer2.value)
		if buffer2.value == 1: # and last_state == 0:
			print("detected")
			time_var_2 = time.time()
			while(time.time() < time_var_2 + 2):
				print("up")
				move.move_drone(speed=[0.0,0.0,0.3])
				#move.move_drone(speed=[0.1,0.0,0.0])
				pass
			
			#move.toggleCam()
			#buffer2.value = 2
			move.move_drone(speed=[0.0,0.0,0.0])
			print("wait")
			time.sleep(7)

		#elif buffer2.value == 1 and last_state == 2:
			time_var_2 = time.time()
			while(time.time() < time_var_2 + 2):
				print("down")
				move.move_drone(speed=[0.0,0.0,-0.3])
				pass
			
			#move.toggleCam()
			move.move_drone(speed=[0.0,0.0,0.0])
			buffer2.value = 0
			
		#last_state = buffer2.value


	time_var = time.time()
	while(time.time() < time_var+3):
		move.land_drone()
        	#buffer.value = 2
		pass
	print("fin process 2")
	move.move_drone(speed=[0.0,0.0,0.0])

def main(args):

	val = Value('i', 0)

	pro1 = Process(target=p1, args=(val,))
	pro2 = Process(target=p2, args=(val,))
	

	pro1.start()
	pro2.start()

	time.sleep(5)

	pro1.join()
	pro2.join()

if __name__ == '__main__':
    main(sys.argv)
