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

class MoveDrone:	
	def __init__(self):
		self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10) 
		# TODO put the takeoff topic name here
		self.landing_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=10)
		# TODO put the landing topic name here
		self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10) # Publish commands to drone 	

	def move_drone(self, speed=[0.0, 0.0, 0.0], orient=[0.0, 0.0, 0.0]):
		vel_msg = Twist()
		# TODO: fill the velocity fields here with the desired values
		vel_msg.linear.x = 0.3
		vel_msg.linear.y = 0.0
		vel_msg.linear.z = 0.0
		# TODO: fill the angulare velocities here with the desired values
		vel_msg.angular.x = orient[0]
		vel_msg.angular.y = orient[1]
		vel_msg.angular.z = orient[2]
		self.move_pub.publish(vel_msg)
		return 0

	def takeoff_drone(self):
		empty_msg = Empty()
		# TODO: send takeoff command to the drone
		#print('i can fly')
		self.takeoff_pub.publish(empty_msg)

	def land_drone(self):
		empty_msg = Empty()
		# TODO: send landing command to the drone
		self.landing_pub.publish(empty_msg)
        
    def toggleCam(self):
       ''' Switches between camera feeds of the AR.Drone '''
       rospy.wait_for_service( 'ardrone/togglecam' )
       try:
           toggle = rospy.ServiceProxy( 'ardrone/togglecam', std_srvs.srv.Empty )
           toggle()
       except rospy.ServiceException, e:
           print "Service call failed: %s"%e


class image_converter:
    
    def __init__(self,link):
          
        #self.image_pub = rospy.Publisher("image_topic_2",Image)
          
        self.bridge = CvBridge()
        self.image = None
        self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",Image,self.callback)
        
        self.face_classifier = cv2.CascadeClassifier('/home/ubuntu/Desktop/drone/src/my_package/scripts/haarcascade_frontalface_default.xml')
          
        self.lien = link

    def set_image_sub(sub_str):
        self.image_sub.unregister()
        self.image_sub = rospy.Subscriber(sub_str,Image,self.callback)
    def callback(self,msg):
        self.image = self.bridge.imgmsg_to_cv2(msg)
        
    def start(self,data):
         while not rospy.is_shutdown(): 
             ros.spinOnce()
             if self.image is not None:
                 cv_image = self.image
                 if self.lien.value == 0
                      
                    #print("im :\n",cv_image,'\n')
                    #print("shape :\n",cv_image.shape,'\n')
                    
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_classifier.detectMultiScale(gray, 1.1, 4)
                    
                    #if faces is ():
                       #print("No Face Found")
                    for (x,y,w,h) in faces:
                        cv2.rectangle(cv_image,(x,y),(x+w,y+h),(127,0,255),2)
                        print("width : {} height : {}".format(w,h))
                        if w > 98:
                            #print("BIPBIP")
                            self.toggleCam()
                            self.set_img_sub("/ardrone/bottom/image_raw")
                            self.lien.value = 1
                        else:
                            self.lien.value = 0
                       
                  if self.lien.value == 2:
                   	 print("hello")
                   	
                       if len(list_pixel) >= 5: #pop
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
                              
                       self.list_pixel.append(pixel)
                       if len(self.list_pixel) >= 5:
                           if(0 in self.list_pixel[0] and 0 in self.list_pixel[1] and 0 in self.list_pixel[2] and 0 in self.list_pixel[3] and 0 in self.list_pixel[4] and self.flag == True):
                               self.flag = False
                   
                           if(0 not in self.list_pixel[0] and 0 not in self.list_pixel[1] and 0 not in self.list_pixel[2] and 0 not in self.list_pixel[3] and 0 not in self.list_pixel[4] and self.flag == False):
                               self.flag = True
                               self.link.value = 3
                               self.toggleCam()
                               self.set_img_sub("/ardrone/front/image_raw")
                       cv2.imshow("Image window", cv_image)
                       cv2.waitKey(1)
                          
            
    def toggleCam(self):
        ''' Switches between camera feeds of the AR.Drone '''
        rospy.wait_for_service( 'ardrone/togglecam' )
        try:
            toggle = rospy.ServiceProxy( 'ardrone/togglecam', std_srvs.srv.Empty )
            toggle()
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    
    

def p1(buffer1):
	ic = image_converter(buffer1)
  	rospy.init_node('image_converter', anonymous=True)
  	while not rospy.is_shutdown():
          
  	cv2.destroyAllWindows()

def p2(buffer2):
	rospy.init_node('basic_controller', anonymous=True)
	move = MoveDrone()
	#move.toggleCam()
	
	print("begin")
	time_var = time.time()
	while(time.time() < time_var+3):
		#move.takeoff_drone()
		pass
	
	#move.move_drone(speed=[0.0,0.0,0.0])

	time_var = time.time()
	while(time.time() < time_var + 3):
		#move.move_drone(speed=[0.0,0.0,0.5])
		pass

	#move.move_drone(speed=[0.0,0.0,0.0])

	#routine
	time_var = time.time()
	time_var_2 = time.time()
	TIME_OF_RUNNING = 30 #seconds

	while (time.time() < time_var+30):
		if buffer2.value == 1 :
			print("test")
			while(time.time() < time_var_2 + 3):
				print("up")
				#move.move_drone(speed=[0.0,0.0,0.2])
				#move.move_drone(speed=[0.1,0.0,0.0])
				pass
			buffer2.value = 2

		elif buffer2.value == 3:

			while(time.time() < time_var_2 + 3):
				#move.move_drone(speed=[0.0,0.0,-0.2])
				pass
			buffer2.value = 0
			

	#move.move_drone(speed=[0.0,0.0,0.0])

	time_var = time.time()
	while(time.time() < time_var+3):
		#move.land_drone()
		pass
	print("fin process 2")
	#move.move_drone(speed=[0.0,0.0,0.0])


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
