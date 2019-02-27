#!/usr/bin/python
import os
import rospy
import numpy as np
from sloth import sloth
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from baxter_gbi_input_msgs.msg import signal

i=8
model_path = os.path.abspath(__file__ + "/../../model/LSTMnet.h5") #my_model6es.h5
c = [10, 10, 10, 10, 10, 10]
tau = [0.75, 0.75, 0.75, 0.75, 0.9, 0.75]
S = sloth(model_path, 40, 6, 3,0.2,tau,c)

global data_stream 
data_stream =  np.empty((1,100,3))
data_stream[:] = np.nan

global gesture_instance
gesture_instance = np.empty((1,100,3))
gesture_instance[:] = np.nan


global last
last = 0

def imu_callback(data):
    global last
    global data_stream
    global gesture_instance
    global S
    #print long(data.header.frame_id) - last
    #if long(data.header.frame_id) - last > 90:
        #print long(data.header.frame_id) - last
    gesture_instance = np.roll(gesture_instance,99,1)
    gesture_instance[:,-1,:] = np.nan
    data_stream = np.roll(data_stream,99,1)
    data_stream[:,-1,0] = data.linear_acceleration.x
    data_stream[:,-1,1] = data.linear_acceleration.y
    data_stream[:,-1,2] = data.linear_acceleration.z
    
    #last = long(data.header.frame_id)
    S.window_update(data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z)
    S.classify()
    S.detect()

def main():
    global data_stream
    global gesture_instance

    rospy.init_node('sloth', anonymous=True)
    rospy.Subscriber('/imu_data', Imu, imu_callback, queue_size=10)
    pub = []
    r = rospy.Rate(10)

    pub.append(rospy.Publisher('smartwatch/wrist_up', signal, queue_size=2))
    pub.append(rospy.Publisher('smartwatch/wrist_down', signal, queue_size=2))
    pub.append(rospy.Publisher('smartwatch/turn_clockwise', signal, queue_size=2))
    pub.append(rospy.Publisher('smartwatch/turn_counterclockwise', signal, queue_size=2))
    pub.append(rospy.Publisher('smartwatch/spike_clockwise', signal, queue_size=2))
    pub.append(rospy.Publisher('smartwatch/spike_counterclockwise', signal, queue_size=2))
    
    gesture_names = ["Wrist Up", "Wrist Down", "Turn Clockwise", "Turn Counter Clockwise",  "Spike Clockwise", "Spike Counter Clockwise"]

    msg = signal()
    msg.device_id = "1"
    msg.device_type = "smartwatch"
    msg.device_model = "G Watch R 5509"
    
    while not rospy.is_shutdown():
        #S.display()
        gesture, probabilities = S.get_gesures()
        i = 0
        for ges in gesture:
            gesture_instance[:,-1,:] = data_stream[:,-1,:]
            #img=mpimg.imread(images_path + "/" + str(ges) + ".jpg")
            #ax1.imshow(img)
            msg.header.stamp = rospy.Time.now()
            msg.action_descr = gesture_names[ges - 1]
            msg.confidence = probabilities[i]
            pub[ges - 1].publish(msg)
            i = i + 1
            
        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
