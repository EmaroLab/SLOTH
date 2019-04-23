#!/usr/bin/python
import os
import rospy
import numpy as np
from sloth import sloth
from sensor_msgs.msg import Imu
from baxter_gbi_input_msgs.msg import signal

i=8
model_path = os.path.abspath(__file__ + "/../../model/LSTMnet.h5") #my_model6es.h5
c = [10, 10, 10, 10, 10, 10]
tau = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
S = sloth(model_path, 30, 6, 3, 0.2, tau, c)

def imu_callback(data):
    global S
    
    #if (rospy.Time.now() - last_timestamp) > rospy.Duration(0.3):
    S.window_update(data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z)
    S.classify()
    S.detect()

def main():
    global last_timestamp

    rospy.init_node('sloth', anonymous=True)
    rospy.Subscriber('/G_Watch_R_5509/imu_data', Imu, imu_callback, queue_size=10)
    pub = []
    r = rospy.Rate(30)

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
    
    last_timestamp = rospy.Time(0)
    while not rospy.is_shutdown():
        #S.display()
        gesture, probabilities = S.get_gestures()
        
        if len(gesture) > 0:
            print("NUM GESTURES: %d" % (len(gesture),))
            ges = gesture[0]
            #img=mpimg.imread(images_path + "/" + str(ges) + ".jpg")
            #ax1.imshow(img)
            msg.header.stamp = rospy.Time.now()
            msg.action_descr = gesture_names[ges - 1]
            msg.confidence = probabilities[0]
            log = "%.3f \t %.5f \t %s" %(msg.header.stamp.to_sec(), probabilities[0], gesture_names[ges - 1])
            if (rospy.Time.now() - last_timestamp) > rospy.Duration(0.5):
                print(log)    
                last_timestamp = msg.header.stamp
                pub[ges - 1].publish(msg)
            else:
                print(log + "\t DISCARDED")

        
        
        
        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
