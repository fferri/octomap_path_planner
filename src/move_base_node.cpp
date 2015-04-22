#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <cstdlib>
#include <cassert>
#include <limits>

#include <boost/foreach.hpp>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/transforms.h>

#include <octomap/octomap.h>
#include <octomap_ros/conversions.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl_conversions/pcl_conversions.h>


template<typename PointA, typename PointB>
double sqdist(const PointA& a, const PointB& b)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}


template<typename PointA, typename PointB>
double dist(const PointA& a, const PointB& b)
{
    return sqrt(sqdist(a, b));
}


class MoveBase
{
protected:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    std::string frame_id_;
    std::string robot_frame_id_;
    ros::Subscriber navfn_sub_;
    ros::Subscriber goal_point_sub_;
    ros::Subscriber goal_pose_sub_;
    ros::Publisher twist_pub_;
    ros::Publisher target_pub_;
    ros::Publisher position_error_pub_;
    ros::Publisher orientation_error_pub_;
    tf::TransformListener tf_listener_;    
    geometry_msgs::PoseStamped robot_pose_;
    geometry_msgs::PoseStamped goal_;
    pcl::PointCloud<pcl::PointXYZI> navfn_;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::Ptr navfn_octree_ptr_;
    ros::Timer controller_timer_;
    double goal_reached_threshold_;
    double controller_frequency_;
    double local_target_radius_;
    double twist_linear_gain_;
    double twist_angular_gain_;
    bool reached_position_;
    void startController();
public:
    MoveBase();
    ~MoveBase();
    void onNavigationFunctionChange(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void onGoal(const geometry_msgs::PointStamped::ConstPtr& msg);
    void onGoal(const geometry_msgs::PoseStamped::ConstPtr& msg);
    int projectPositionToNavigationFunction(const geometry_msgs::Point& pos);
    void projectGoalPositionToNavigationFunction();
    bool getRobotPose();
    double positionError();
    double orientationError();
    bool generateLocalTarget(geometry_msgs::PointStamped& p_local);
    void generateTwistCommand(const geometry_msgs::PointStamped& local_target, geometry_msgs::Twist& twist);
    void controllerCallback(const ros::TimerEvent& event);
};


MoveBase::MoveBase()
    : pnh_("~"),
      frame_id_("/map"),
      robot_frame_id_("/base_link"),
      goal_reached_threshold_(0.2),
      controller_frequency_(2.0),
      local_target_radius_(0.4),
      twist_linear_gain_(0.5),
      twist_angular_gain_(1.0),
      reached_position_(false)
{
    pnh_.param("frame_id", frame_id_, frame_id_);
    pnh_.param("robot_frame_id", robot_frame_id_, robot_frame_id_);
    pnh_.param("goal_reached_threshold", goal_reached_threshold_, goal_reached_threshold_);
    pnh_.param("controller_frequency", controller_frequency_, controller_frequency_);
    pnh_.param("local_target_radius", local_target_radius_, local_target_radius_);
    pnh_.param("twist_linear_gain", twist_linear_gain_, twist_linear_gain_);
    pnh_.param("twist_angular_gain", twist_angular_gain_, twist_angular_gain_);
    navfn_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>("navfn_in", 1, &MoveBase::onNavigationFunctionChange, this);
    goal_point_sub_ = nh_.subscribe<geometry_msgs::PointStamped>("goal_point_in", 1, &MoveBase::onGoal, this);
    goal_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("goal_pose_in", 1, &MoveBase::onGoal, this);
    twist_pub_ = nh_.advertise<geometry_msgs::Twist>("twist_out", 1, false);
    target_pub_ = nh_.advertise<geometry_msgs::PointStamped>("target_out", 1, false);
    position_error_pub_ = nh_.advertise<std_msgs::Float32>("position_error", 10, false);;
    orientation_error_pub_ = nh_.advertise<std_msgs::Float32>("orientation_error", 10, false);;
}


MoveBase::~MoveBase()
{
}


void MoveBase::onNavigationFunctionChange(const sensor_msgs::PointCloud2::ConstPtr& pcl_msg)
{
    pcl::PointCloud<pcl::PointXYZI> pcl;
    pcl::fromROSMsg(*pcl_msg, pcl);

    navfn_.clear();
    pcl_ros::transformPointCloud(frame_id_, pcl, navfn_, tf_listener_);

    navfn_octree_ptr_ = pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::Ptr(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>(0.01));
    navfn_octree_ptr_->setInputCloud(navfn_.makeShared());
    navfn_octree_ptr_->addPointsFromInputCloud();
}


void MoveBase::startController()
{
    controller_timer_ = nh_.createTimer(ros::Duration(1.0 / controller_frequency_), &MoveBase::controllerCallback, this);
    reached_position_ = false;
}


void MoveBase::onGoal(const geometry_msgs::PointStamped::ConstPtr& msg)
{
    try
    {
        geometry_msgs::PointStamped msg2;
        tf_listener_.transformPoint(frame_id_, *msg, msg2);
        goal_.header.stamp = msg2.header.stamp;
        goal_.header.frame_id = msg2.header.frame_id;
        goal_.pose.position.x = msg2.point.x;
        goal_.pose.position.y = msg2.point.y;
        goal_.pose.position.z = msg2.point.z;
        goal_.pose.orientation.x = 0.0;
        goal_.pose.orientation.y = 0.0;
        goal_.pose.orientation.z = 0.0;
        goal_.pose.orientation.w = 0.0;
        projectGoalPositionToNavigationFunction();
        ROS_INFO("goal set to point (%f, %f, %f)",
            goal_.pose.position.x, goal_.pose.position.y, goal_.pose.position.z);

        startController();
    }
    catch(tf::TransformException& ex)
    {
        ROS_ERROR("Failed to lookup robot position: %s", ex.what());
    }
}


void MoveBase::onGoal(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    try
    {
        tf_listener_.transformPose(frame_id_, *msg, goal_);
        projectGoalPositionToNavigationFunction();
        ROS_INFO("goal set to pose (%f, %f, %f), (%f, %f, %f, %f)",
                goal_.pose.position.x, goal_.pose.position.y, goal_.pose.position.z,
                goal_.pose.orientation.x, goal_.pose.orientation.y, goal_.pose.orientation.z,
                goal_.pose.orientation.w);

        startController();
    }
    catch(tf::TransformException& ex)
    {
        ROS_ERROR("Failed to lookup robot position: %s", ex.what());
    }
}


int MoveBase::projectPositionToNavigationFunction(const geometry_msgs::Point& pos)
{
    pcl::PointXYZI p;
    p.x = pos.x;
    p.y = pos.y;
    p.z = pos.z;
    std::vector<int> pointIdx;
    std::vector<float> pointDistSq;
    if(navfn_octree_ptr_->nearestKSearch(goal, 1, pointIdx, pointDistSq) < 1)
    {
        return -1;
    }
    return pointIdx[0];
}


void MoveBase::projectGoalPositionToNavigationFunction()
{
    int goal_index = projectPositionToNavigationFunction(goal_.pose.position);
    if(goal_index == -1)
    {
        ROS_ERROR("Failed to project goal position to navfn pcl");
        return;
    }
    goal_.pose.position.x = navfn_[goal_index].x;
    goal_.pose.position.y = navfn_[goal_index].y;
    goal_.pose.position.z = navfn_[goal_index].z;
}


bool MoveBase::getRobotPose()
{
    try
    {
        geometry_msgs::PoseStamped robot_pose_local;
        robot_pose_local.header.frame_id = robot_frame_id_;
        robot_pose_local.pose.position.x = 0.0;
        robot_pose_local.pose.position.y = 0.0;
        robot_pose_local.pose.position.z = 0.0;
        robot_pose_local.pose.orientation.x = 0.0;
        robot_pose_local.pose.orientation.y = 0.0;
        robot_pose_local.pose.orientation.z = 0.0;
        robot_pose_local.pose.orientation.w = 1.0;
        tf_listener_.transformPose(frame_id_, robot_pose_local, robot_pose_);
        return true;
    }
    catch(tf::TransformException& ex)
    {
        ROS_ERROR("Failed to lookup robot position: %s", ex.what());
    }
}


double MoveBase::positionError()
{
    return sqrt(
            pow(robot_pose_.pose.position.x - goal_.pose.position.x, 2) +
            pow(robot_pose_.pose.position.y - goal_.pose.position.y, 2) +
            pow(robot_pose_.pose.position.z - goal_.pose.position.z, 2)
    );
}


double MoveBase::orientationError()
{
    // check if goal is only by position:
    double qnorm = pow(goal_.pose.orientation.w, 2) +
            pow(goal_.pose.orientation.x, 2) +
            pow(goal_.pose.orientation.y, 2) +
            pow(goal_.pose.orientation.z, 2);

    // if so, we never have an orientation error:
    if(qnorm < 1e-5) return 0;

    // "Robotica - Modellistica Pianificazione e Controllo" eq. 3.88
    double nd = goal_.pose.orientation.w,
            ne = robot_pose_.pose.orientation.w;
    Eigen::Vector3d ed(goal_.pose.orientation.x, goal_.pose.orientation.y, goal_.pose.orientation.z),
            ee(robot_pose_.pose.orientation.x, robot_pose_.pose.orientation.y, robot_pose_.pose.orientation.z);
    Eigen::Vector3d eo = ne * ed - nd * ee - ed.cross(ee);
    return eo(2);
}


bool MoveBase::generateLocalTarget(geometry_msgs::PointStamped& p_local)
{
    int min_index = -1, min_index_straight = -1;
    float min_value = std::numeric_limits<float>::infinity();

    pcl::PointXYZI robot_position;
    robot_position.x = robot_pose_.pose.position.x;
    robot_position.y = robot_pose_.pose.position.y;
    robot_position.z = robot_pose_.pose.position.z;

    std::vector<int> pointIdx;
    std::vector<float> pointDistSq;
    navfn_octree_ptr_->radiusSearch(robot_position, local_target_radius_, pointIdx, pointDistSq);
    pcl::PointCloud<pcl::PointXYZI> neighbors, neighbors_local;
    neighbors.header.frame_id = navfn_.header.frame_id;
    neighbors.header.stamp = navfn_.header.stamp;
    for(std::vector<int>::iterator it = pointIdx.begin(); it != pointIdx.end(); ++it)
        neighbors.push_back(navfn_[*it]);

    if(!pcl_ros::transformPointCloud(robot_frame_id_, neighbors, neighbors_local, tf_listener_))
    {
        ROS_ERROR("Failed to transform robot neighborhood");
        return false;
    }

    for(size_t i = 0; i < neighbors_local.size(); i++)
    {
        pcl::PointXYZI& p = neighbors_local[i];

        if(p.intensity < min_value)
        {
            min_value = p.intensity;
            min_index = i;
        }
    }

    if(min_index == -1)
    {
        ROS_ERROR("Failed to find a target in robot vicinity");
        return false;
    }

    p_local.header.stamp = ros::Time::now();
    p_local.header.frame_id = neighbors_local.header.frame_id;
    p_local.point.x = neighbors_local[min_index].x;
    p_local.point.y = neighbors_local[min_index].y;
    p_local.point.z = neighbors_local[min_index].z;

    target_pub_.publish(p_local);

    return true;
}


void MoveBase::generateTwistCommand(const geometry_msgs::PointStamped& local_target, geometry_msgs::Twist& twist)
{
    if(local_target.header.frame_id != robot_frame_id_)
    {
        ROS_ERROR("generateTwistCommand: local_target must be in frame '%s'", robot_frame_id_.c_str());
        return;
    }

    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.linear.z = 0.0;
    twist.angular.x = 0.0;
    twist.angular.y = 0.0;
    twist.angular.z = 0.0;

    const geometry_msgs::Point& p = local_target.point;

    if(p.x < 0 || fabs(p.y) > p.x)
    {
        // turn in place
        twist.angular.z = (p.y > 0 ? 1 : -1) * twist_angular_gain_;
    }
    else
    {
        // make arc
        double center_y = (pow(p.x, 2) + pow(p.y, 2)) / (2 * p.y);
        double theta = fabs(atan2(p.x, fabs(center_y) - fabs(p.y)));
        double arc_length = fabs(center_y * theta);

        twist.linear.x = twist_linear_gain_ * arc_length;
        twist.angular.z = twist_angular_gain_ * (p.y >= 0 ? 1 : -1) * theta;
    }
}


void MoveBase::controllerCallback(const ros::TimerEvent& event)
{
    if(!getRobotPose())
    {
        ROS_ERROR("controllerCallback: failed to get robot pose");
        return;
    }

    geometry_msgs::Twist twist;
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.linear.z = 0.0;
    twist.angular.x = 0.0;
    twist.angular.y = 0.0;
    twist.angular.z = 0.0;

    std_msgs::Float32 ep, eo;

    ep.data = positionError();
    position_error_pub_.publish(ep);

    eo.data = orientationError();
    orientation_error_pub_.publish(eo);

    const char *status_str;

    if((!reached_position_ && ep.data > goal_reached_threshold_)
        || (reached_position_ && ep.data > 2 * goal_reached_threshold_))
    {
        // regulate position

        status_str = "REGULATING POSITION";

        reached_position_ = false;

        geometry_msgs::PointStamped local_target;

        if(!generateLocalTarget(local_target))
        {
            ROS_ERROR("controllerCallback: failed to generate a local target to follow");
            return;
        }

        generateTwistCommand(local_target, twist);
    }
    else
    {
        reached_position_ = true;

        if(fabs(eo.data) > 0.02)
        {
            // regulate orientation

            status_str = "REGULATING ORIENTATION";

            twist.angular.z = twist_angular_gain_ * eo.data;
        }
        else
        {
            // goal reached

            status_str = "REACHED GOAL";

            ROS_INFO("goal reached! stopping controller timer");

            controller_timer_.stop();
        }
    }

    ROS_INFO("controller: ep=%f, eo=%f, status=%s", ep.data, eo.data, status_str);

    twist_pub_.publish(twist);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "move_base");

    MoveBase p;
    ros::spin();

    return 0;
}

