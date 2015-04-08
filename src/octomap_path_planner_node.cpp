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
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>

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

class OctomapPathPlanner
{
protected:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    std::string frame_id_;
    std::string robot_frame_id_;
    ros::Subscriber octree_sub_;
    ros::Subscriber goal_sub_;
    ros::Publisher ground_pub_;
    ros::Publisher obstacles_pub_;
    ros::Publisher path_pub_;
    ros::Publisher twist_pub_;
    ros::Publisher target_pub_;
    tf::TransformListener tf_listener_;    
    geometry_msgs::PoseStamped robot_pose_;
    geometry_msgs::PointStamped goal_;
    octomap::OcTree* octree_ptr_;
    pcl::PointCloud<pcl::PointXYZI> ground_pcl_;
    pcl::PointCloud<pcl::PointXYZ> obstacles_pcl_;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::Ptr ground_octree_ptr_;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr obstacles_octree_ptr_;
    ros::Timer controller_timer_;
    bool treat_unknown_as_free_;
    double robot_height_;
    double robot_radius_;
    double goal_reached_threshold_;
    double controller_frequency_;
    double local_target_radius_;
    double twist_linear_gain_;
    double twist_angular_gain_;
    double max_superable_height_;
public:
    OctomapPathPlanner();
    ~OctomapPathPlanner();
    void onOctomap(const octomap_msgs::Octomap::ConstPtr& msg);
    void onGoal(const geometry_msgs::PointStamped::ConstPtr& msg);
    void expandOcTree();
    bool isGround(const octomap::OcTreeKey& key);
    bool isObstacle(const octomap::OcTreeKey& key);
    void computeGround();
    void publishGroundCloud();
    void computeDistanceTransform();
    bool getRobotPose();
    bool goalReached();
    int generateTarget();
    bool generateLocalTarget(geometry_msgs::PointStamped& p_local);
    void generateTwistCommand(const geometry_msgs::PointStamped& local_target, geometry_msgs::Twist& twist);
    void controllerCallback(const ros::TimerEvent& event);
};


OctomapPathPlanner::OctomapPathPlanner()
    : pnh_("~"),
      frame_id_("/map"),
      robot_frame_id_("/base_link"),
      octree_ptr_(0L),
      treat_unknown_as_free_(false),
      robot_height_(0.5),
      robot_radius_(0.5),
      goal_reached_threshold_(0.2),
      controller_frequency_(2.0),
      local_target_radius_(0.4),
      twist_linear_gain_(0.5),
      twist_angular_gain_(1.0),
      max_superable_height_(0.2)
{
    pnh_.param("frame_id", frame_id_, frame_id_);
    pnh_.param("robot_frame_id", robot_frame_id_, robot_frame_id_);
    pnh_.param("treat_unknown_as_free", treat_unknown_as_free_, treat_unknown_as_free_);
    pnh_.param("robot_height", robot_height_, robot_height_);
    pnh_.param("robot_radius", robot_radius_, robot_radius_);
    pnh_.param("goal_reached_threshold", goal_reached_threshold_, goal_reached_threshold_);
    pnh_.param("controller_frequency", controller_frequency_, controller_frequency_);
    pnh_.param("local_target_radius", local_target_radius_, local_target_radius_);
    pnh_.param("twist_linear_gain", twist_linear_gain_, twist_linear_gain_);
    pnh_.param("twist_angular_gain", twist_angular_gain_, twist_angular_gain_);
    pnh_.param("max_superable_height", max_superable_height_, max_superable_height_);
    octree_sub_ = nh_.subscribe<octomap_msgs::Octomap>("octree_in", 1, &OctomapPathPlanner::onOctomap, this);
    goal_sub_ = nh_.subscribe<geometry_msgs::PointStamped>("goal_in", 1, &OctomapPathPlanner::onGoal, this);
    ground_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("ground_cloud_out", 1, true);
    obstacles_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obstacles_cloud_out", 1, true);
    path_pub_ = nh_.advertise<nav_msgs::Path>("path_out", 1, true);
    twist_pub_ = nh_.advertise<geometry_msgs::Twist>("twist_out", 1, false);
    target_pub_ = nh_.advertise<geometry_msgs::PointStamped>("target_out", 1, false);
    ground_pcl_.header.frame_id = frame_id_;
    obstacles_pcl_.header.frame_id = frame_id_;
}


OctomapPathPlanner::~OctomapPathPlanner()
{
    if(octree_ptr_) delete octree_ptr_;
}


void OctomapPathPlanner::onOctomap(const octomap_msgs::Octomap::ConstPtr& msg)
{
    if(octree_ptr_) delete octree_ptr_;
    octree_ptr_ = octomap_msgs::binaryMsgToMap(*msg);

    expandOcTree();
    computeGround();
    computeDistanceTransform();
}


void OctomapPathPlanner::onGoal(const geometry_msgs::PointStamped::ConstPtr& msg)
{
    try
    {
        tf_listener_.transformPoint(frame_id_, *msg, goal_);
        ROS_INFO("goal set to position (%f, %f, %f)", goal_.point.x, goal_.point.y, goal_.point.z);

        controller_timer_ = nh_.createTimer(ros::Duration(1.0 / controller_frequency_), &OctomapPathPlanner::controllerCallback, this);
    }
    catch(tf::TransformException& ex)
    {
        ROS_ERROR("Failed to lookup robot position: %s", ex.what());
    }

    computeDistanceTransform();
}


void OctomapPathPlanner::expandOcTree()
{
    if(!octree_ptr_) return;

    ROS_INFO("begin expanding octree (octree size = %ld)", octree_ptr_->size());

    unsigned int maxDepth = octree_ptr_->getTreeDepth();

    // expand collapsed occupied nodes until all occupied leaves are at maximum depth
    std::vector<octomap::OcTreeNode*> collapsed_occ_nodes;

    do
    {
        collapsed_occ_nodes.clear();
        for(octomap::OcTree::iterator it = octree_ptr_->begin(); it != octree_ptr_->end(); ++it)
        {
            if(octree_ptr_->isNodeOccupied(*it) && it.getDepth() < maxDepth)
            {
                collapsed_occ_nodes.push_back(&(*it));
            }
        }
        for(std::vector<octomap::OcTreeNode*>::iterator it = collapsed_occ_nodes.begin(); it != collapsed_occ_nodes.end(); ++it)
        {
            (*it)->expandNode();
        }
        ROS_INFO("expanded %ld nodes in octree", collapsed_occ_nodes.size());
    } while(collapsed_occ_nodes.size() > 0);

    ROS_INFO("finished expanding octree");
}


bool OctomapPathPlanner::isGround(const octomap::OcTreeKey& key)
{
    octomap::OcTreeNode *node = octree_ptr_->search(key);
    if(!node) return false;
    if(!octree_ptr_->isNodeOccupied(node)) return false;

    double res = octree_ptr_->getResolution();
    int steps = ceil(robot_height_ / res);
    octomap::OcTreeKey key1(key);
    while(steps-- > 0)
    {
        key1[2]++;
        node = octree_ptr_->search(key1);
        if(!node)
        {
            if(!treat_unknown_as_free_) return false;
        }
        else
        {
            if(octree_ptr_->isNodeOccupied(node)) return false;
        }
    }
    return true;
}


bool OctomapPathPlanner::isObstacle(const octomap::OcTreeKey& key)
{
    octomap::OcTreeNode *node;
    double res = octree_ptr_->getResolution();
    int num_voxels = 1;

    // look up...
    octomap::OcTreeKey key1(key);
    while(true)
    {
        key1[2]++;
        node = octree_ptr_->search(key1);
        if(!node) break;
        if(node && !octree_ptr_->isNodeOccupied(node)) break;
        num_voxels++;
    }

    // look down...
    octomap::OcTreeKey key2(key);
    while(true)
    {
        key2[2]--;
        node = octree_ptr_->search(key2);
        if(!node) break;
        if(node && !octree_ptr_->isNodeOccupied(node)) break;
        num_voxels++;
    }

    return res * num_voxels > max_superable_height_;
}


void OctomapPathPlanner::computeGround()
{
    if(!octree_ptr_) return;

    ROS_INFO("begin computing ground");

    ground_pcl_.clear();
    obstacles_pcl_.clear();

    for(octomap::OcTree::leaf_iterator it = octree_ptr_->begin(); it != octree_ptr_->end(); ++it)
    {
        if(!octree_ptr_->isNodeOccupied(*it)) continue;

        if(isGround(it.getKey()))
        {
            pcl::PointXYZI point;
            point.x = it.getX();
            point.y = it.getY();
            point.z = it.getZ();
            point.intensity = std::numeric_limits<float>::infinity();
            ground_pcl_.push_back(point);
        }
        else if(isObstacle(it.getKey()))
        {
            pcl::PointXYZ point;
            point.x = it.getX();
            point.y = it.getY();
            point.z = it.getZ();
            obstacles_pcl_.push_back(point);
        }
    }

    double res = octree_ptr_->getResolution();

    ground_octree_ptr_ = pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::Ptr(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>(res));
    ground_octree_ptr_->setInputCloud(ground_pcl_.makeShared());
    ground_octree_ptr_->addPointsFromInputCloud();

    obstacles_octree_ptr_ = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(res));
    obstacles_octree_ptr_->setInputCloud(obstacles_pcl_.makeShared());
    obstacles_octree_ptr_->addPointsFromInputCloud();

    ROS_INFO("finished computing ground");
}


void OctomapPathPlanner::publishGroundCloud()
{
    if(ground_pub_.getNumSubscribers() > 0)
    {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(ground_pcl_, msg);
        ground_pub_.publish(msg);
    }

    if(obstacles_pub_.getNumSubscribers() > 0)
    {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(obstacles_pcl_, msg);
        obstacles_pub_.publish(msg);
    }
}


void OctomapPathPlanner::computeDistanceTransform()
{
    if(ground_pcl_.size() == 0)
    {
        ROS_INFO("skip computing distance transform because ground_pcl_ is empty");
        return;
    }

    ROS_INFO("begin computing distance transform (ground pcl size = %ld)", ground_pcl_.size());

    // find goal index in ground pcl:
    std::vector<int> pointIdx;
    std::vector<float> pointDistSq;
    pcl::PointXYZI goal;
    goal.x = goal_.point.x;
    goal.y = goal_.point.y;
    goal.z = goal_.point.z;
    if(ground_octree_ptr_->nearestKSearch(goal, 1, pointIdx, pointDistSq) < 1)
    {
        ROS_ERROR("unable to find goal in ground pcl");
        return;
    }

    double res = octree_ptr_->getResolution();
    int goal_idx = pointIdx[0];

    // distance to goal is zero (stored in the intensity channel):
    ground_pcl_[goal_idx].intensity = 0.0;

    std::queue<int> q;
    q.push(goal_idx);
    while(!q.empty())
    {
        int i = q.front();
        q.pop();

        // get neighbours:
        std::vector<int> pointIdx;
        std::vector<float> pointDistSq;
        ground_octree_ptr_->radiusSearch(ground_pcl_[i], 1.8 * res, pointIdx, pointDistSq);

        for(std::vector<int>::iterator it = pointIdx.begin(); it != pointIdx.end(); it++)
        {
            int j = *it;

            // intensity value are initially set to infinity.
            // if i is finite it means it has already been labeled.
            if(std::isfinite(ground_pcl_[j].intensity)) continue;

            // otherwise, label it:
            ground_pcl_[j].intensity = ground_pcl_[i].intensity + 1.0;

            // continue exploring neighbours:
            q.push(j);
        }
    }

    // normalize intensity:
    float imin = std::numeric_limits<float>::infinity();
    float imax = -std::numeric_limits<float>::infinity();
    for(pcl::PointCloud<pcl::PointXYZI>::iterator it = ground_pcl_.begin(); it != ground_pcl_.end(); ++it)
    {
        if(!std::isfinite(it->intensity)) continue;
        imin = fmin(imin, it->intensity);
        imax = fmax(imax, it->intensity);
    }
    ROS_INFO("intensity bounds before normalization: <%f, %f>", imin, imax);
    const float eps = 0.01;
    float d = imax - imin + eps;
    for(pcl::PointCloud<pcl::PointXYZI>::iterator it = ground_pcl_.begin(); it != ground_pcl_.end(); ++it)
    {
        if(std::isfinite(it->intensity))
            it->intensity = (it->intensity - imin) / d;
        else
            it->intensity = 1.0;
    }

    publishGroundCloud();

    ROS_INFO("finished computing distance transform");
}


bool OctomapPathPlanner::getRobotPose()
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


bool OctomapPathPlanner::goalReached()
{
    double dist = sqrt(
            pow(robot_pose_.pose.position.x - goal_.point.x, 2) +
            pow(robot_pose_.pose.position.y - goal_.point.y, 2) +
            pow(robot_pose_.pose.position.z - goal_.point.z, 2)
    );

    return dist <= goal_reached_threshold_;
}


int OctomapPathPlanner::generateTarget()
{
    int best_index = -1;
    float best_value = std::numeric_limits<float>::infinity();

    pcl::PointXYZI robot_position;
    robot_position.x = robot_pose_.pose.position.x;
    robot_position.y = robot_pose_.pose.position.y;
    robot_position.z = robot_pose_.pose.position.z;

    for(int k = 1; k <= 2; k++)
    {
        std::vector<int> pointIdx;
        std::vector<float> pointDistSq;
        ground_octree_ptr_->radiusSearch(robot_position, k * local_target_radius_, pointIdx, pointDistSq);

        for(std::vector<int>::iterator it = pointIdx.begin(); it != pointIdx.end(); ++it)
        {
            pcl::PointXYZI& p = ground_pcl_[*it];

            std::vector<int> pointIdx2;
            std::vector<float> pointDistSq2;
            pcl::PointXYZ p1; p1.x = p.x; p1.y = p.y; p1.z = p.z;
            bool safe = obstacles_octree_ptr_->nearestKSearch(p1, 1, pointIdx2, pointDistSq2) < 1 || pointDistSq2[0] > (robot_radius_ * robot_radius_);

            if(!safe) continue;

            if(best_index == -1 || p.intensity < best_value)
            {
                best_value = p.intensity;
                best_index = *it;
            }
        }

        if(best_index != -1)
            return best_index;
    }

    return -1;
}


bool OctomapPathPlanner::generateLocalTarget(geometry_msgs::PointStamped& p_local)
{
    int i = generateTarget();

    if(i == -1)
    {
        ROS_ERROR("Failed to find a target in robot vicinity");
        return false;
    }

    try
    {
        geometry_msgs::PointStamped p;
        p.header.frame_id = ground_pcl_.header.frame_id;
        p.point.x = ground_pcl_[i].x;
        p.point.y = ground_pcl_[i].y;
        p.point.z = ground_pcl_[i].z;
        tf_listener_.transformPoint(robot_frame_id_, p, p_local);
        target_pub_.publish(p);
        return true;
    }
    catch(tf::TransformException& ex)
    {
        ROS_ERROR("Failed to transform reference point: %s", ex.what());
        return false;
    }
}


void OctomapPathPlanner::generateTwistCommand(const geometry_msgs::PointStamped& local_target, geometry_msgs::Twist& twist)
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


void OctomapPathPlanner::controllerCallback(const ros::TimerEvent& event)
{
    ROS_INFO("controller callback!");

    if(!getRobotPose())
    {
        ROS_ERROR("controllerCallback: failed to get robot pose");
        return;
    }

    if(goalReached())
    {
        ROS_INFO("goal reached! stopping controller timer");
        controller_timer_.stop();
        geometry_msgs::Twist twist;
        twist.linear.x = 0.0;
        twist.linear.y = 0.0;
        twist.linear.z = 0.0;
        twist.angular.x = 0.0;
        twist.angular.y = 0.0;
        twist.angular.z = 0.0;
        twist_pub_.publish(twist);
        return;
    }

    geometry_msgs::PointStamped local_target;

    if(!generateLocalTarget(local_target))
    {
        ROS_ERROR("controllerCallback: failed to generate a local target to follow");
        return;
    }

    geometry_msgs::Twist twist;

    generateTwistCommand(local_target, twist);

    twist_pub_.publish(twist);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "octomap_path_planner");

    OctomapPathPlanner p;
    ros::spin();

    return 0;
}

