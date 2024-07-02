import rospy
import re
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time
from std_msgs.msg import String

time_data, current_data = [], [] 
t_start = time.time()
figure = plt.figure()
current_plot = figure.add_subplot(1, 1, 1)
current_plot.set_ylim(-1.75, 1.75)

def update_graph(frame):
    current_plot.clear()
    current_plot.plot(time_data, current_data)
    return current_plot

def callback_fn(msg):
    current = float(re.match(r".*Iq:(-*\d+\.\d+)", msg.data).group(1))
    time_data.append(time.time() - t_start)
    current_data.append(current)
    print(current_data[-1])

def listener():
    rospy.init_node("current_listener", anonymous=True)
    rospy.Subscriber("/bariflex", String, callback_fn)
    
    animation = anim.FuncAnimation(figure, update_graph, interval=200)
    plt.show()  

    rospy.spin()

if __name__ == "__main__":
    listener()