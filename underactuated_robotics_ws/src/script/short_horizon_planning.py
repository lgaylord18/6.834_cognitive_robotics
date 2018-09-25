#we have not yet had the opportunity to test the code contained in this file
#on the robots or in the simulator due to issues with the VM
#some of the ROS interactions may not be perfect

#necessary ROS packages
import rospy
import tf
from nav_msgs.msg import *
from geometry_msgs.msg import *

#data handling and path planning function call
import numpy as np
from math import isclose
from lqr_rrt_star import lqr_rrt_star

#turns waypoints from list of [x,y,x,y] to [(x,y), (x,y)]
def split_list(current_list, coordinate_list):
    coordinate_list = [];
    it = iter(current_list)
    points = zip(it, it)
    for point in points:
        coordinate_list.append(point)
    return coordinate_list

def short_horizon_planning(MAPP1_info, SLAM_map):
    #divide waypoints into sections for each robot
    MAPP_full_waypoints = {};
    robot_num = 0;
    index = 0;
    for value in MAPP1_info.robots:
        if (index+value) < len(MAPP1_info.paths):
            MAPP_full_waypoints[robot_num] = MAPP1_info.paths[index:value+1];
        else:
            MAPP_full_waypoints[robot_num] = MAPP1_info.paths[index:]
        index += value;
        robot_num += 1;

    #selects the proper waypoints for the current robot
    MAPP_waypoints = []; #parameter may require tuning
    threshold = 0.05;
    for robot in MAPP_full_waypoints:
        initial_pos = (MAPP_full_waypoints[robot][0], MAPP_full_waypoints[robot][1])
        #check that two points are close - not exact equality due to drift
        if isclose(initial_pos[0], current_pos[0], threshold) and isclose(initial_pos[1], current_pos[1], threshold):
            MAPP_waypoints = split_list(MAPP_full_waypoints[robot])
    goal = MAPP_waypoints[-1]; 

    #iterate through SLAM map to create numpy array
    OURmap=[]
    for i in range(gridY):
        row=[]
        for j in range(gridX):
            val=SLAMmap[i+j]
            if val==100:
                row.append(1)
            elif val==0:
                row.append(0)
            else:
                row.append(-1)
        OURmap.append(row)
    obstacle_map=np.array(OURmap)

    waypoints = lqr_rrt_star(obstacle_map, current_pos, goal);

    #converts waypoints from inertial frame to robot frame
    pose_waypoints = [];
    try:
        (trans,rot) = listener.lookupTransform('/base_link', '/odom', rospy.Time(0));
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue;
    for point in waypoints:
        x = point[0] + trans.x;
        y = point[1] + trans.y;
        pose_waypoints.append((x,y));
    return pose_waypoints;

#captures current position from odometry topic
#converts position from robot frame to intertial frame
def position_capture(msg):
    global current_pos;
    xpos = msg.pose.x;
    ypos = msg.pose.y;
    try:
        (trans,rot) = listener.lookupTransform('/odom', '/base_link', rospy.Time(0));
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue;
    current_pos = (xpos+trans.x, ypos+trans.y);

#captures SLAM map from map topic
def map_capture(msg):
    global current_map;
    current_map = msg.data;

if __name__ == "main":
    rospy.init_node('odometry', anonymous=True);
    odom_sub = rospy.Subscriber('odom',Odometry, position_capture);

    rospy.init_node('map', anonymous= True);
    map_sub = rospy.Subscriber('map',OccupancyGrid, map_capture);

    #calls service provided by MAPP1
    rospy.wait_for_service('get_mapf_paths');
    get_mapf_paths = rospy.ServiceProxy('get_mapf_paths', MapfPaths);
    MAPP1_info = get_mapf_paths();

    global listener
    listener = tf.TransformListener();
    
    waypoints = short_horizon_planning(MAPP1_info, current_map);

    #publishes final waypoints to move_base
    move_base_pub = rospy.Publisher("/move_base_simple", PoseStamped, queue_size = 1)
    rate = rospy.rate(100) #100Hz
    for point in waypoints:
        move_point = PoseStamped();
        move_point.Pose.position.x = point[0];
        move_point.Pose.position.y = point[1];
        move_base_pub.publish(move_point);
        rate.sleep();

    rospy.spin();
