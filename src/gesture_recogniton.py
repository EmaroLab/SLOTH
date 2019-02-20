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
model_path = os.path.abspath(__file__ + "/../../model/conv_net/my_model"+str(i)+".h5") #my_model6es.h5
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
    if long(data.header.frame_id) - last > 90:
        #print long(data.header.frame_id) - last
        gesture_instance = np.roll(gesture_instance,99,1)
        gesture_instance[:,-1,:] = np.nan
        data_stream = np.roll(data_stream,99,1)
        data_stream[:,-1,0] = data.linear_acceleration.x
        data_stream[:,-1,1] = data.linear_acceleration.y
        data_stream[:,-1,2] = data.linear_acceleration.z
        
        last = long(data.header.frame_id)
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
    plt.ion()

    gridsize = (3, 2)
    fig = plt.figure(figsize=(40, 16))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=3)
    ax2 = plt.subplot2grid(gridsize, (0, 1))
    ax3 = plt.subplot2grid(gridsize, (1, 1))
    ax4 = plt.subplot2grid(gridsize, (2, 1))

    ax1.set_xticks([])
    ax1.set_yticks([])

    ticks_label = ["", "0", "2", "4", "6", "8", "10", ""]

    images_path = os.path.abspath(__file__ + "/../../gestures_images/")
    
	msg = signal()
	msg.device_id = "1"
	msg.device_type = "smartwatch"
	msg.device_model = "G Watch R 5509"
    
    while not rospy.is_shutdown():
        #S.display()
        gesture, probabilities = S.get_gesures()
        i = 0
        for ges in gesture:
            print ges
            gesture_instance[:,-1,:] = data_stream[:,-1,:]
            img=mpimg.imread(images_path + "/" + str(ges) + ".jpg")
            ax1.imshow(img)
            msg.header.stamp = rospy.Time.now()
            msg.action_descr = gesture_names[ges - 1]
            msg.confidence = probabilities[i]
            pub[ges - 1].publish(msg)
            i = i + 1
            

        ax2.clear()
        ax3.clear()
        ax4.clear()

        ax2.set_ylim([-14,14])
        ax3.set_ylim([-14,14])
        ax4.set_ylim([-14,14])

        ax2.set_xlim([-20,120])
        ax3.set_xlim([-20,120])
        ax4.set_xlim([-20,120])

        ax2.set_ylabel("Linear Acc X (m/s^2)")
        ax3.set_ylabel("Linear Acc Y (m/s^2)")
        ax4.set_ylabel("Linear Acc Z (m/s^2)")

        ax2.set_xlabel("Time (s)")
        ax3.set_xlabel("Time (s)")
        ax4.set_xlabel("Time (s)")

        ax2.set_xticklabels(ticks_label)
        ax3.set_xticklabels(ticks_label)
        ax4.set_xticklabels(ticks_label)

        ax2.scatter(range(0,100), gesture_instance[0,:,0],s=100 ,c="g")
        ax3.scatter(range(0,100), gesture_instance[0,:,1],s=100, c="g")
        ax4.scatter(range(0,100), gesture_instance[0,:,2],s=100, c="g")

        ax2.plot(range(0,100),data_stream[0,:,0])
        ax3.plot(range(0,100),data_stream[0,:,1])
        ax4.plot(range(0,100),data_stream[0,:,2])
        fig.canvas.flush_events()
        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
