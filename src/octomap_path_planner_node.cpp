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
    std::string global_frame_id_;
    std::string robot_frame_id_;
    ros::Subscriber octree_sub_;
    ros::Subscriber goal_sub_;
    ros::Publisher ground_pub_;
    ros::Publisher path_pub_;
    tf::TransformListener tf_listener_;    
    geometry_msgs::PoseStamped robot_pose_;
    geometry_msgs::PoseStamped target_pose_;
    octomap::OcTree* octree_ptr_;
    pcl::PointCloud<pcl::PointXYZI> ground_pcl_;
    bool treat_unknown_as_free_;
    double robot_height_;
public:
    OctomapPathPlanner();
    ~OctomapPathPlanner();
    void onOctomap(const octomap_msgs::Octomap::ConstPtr& msg);
    void onGoal(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void expandOcTree();
    bool isGround(const octomap::OcTreeKey& key);
    void computeGround();
    void publishGroundCloud();
    void computeDistanceTransform();
};


OctomapPathPlanner::OctomapPathPlanner()
    : pnh_("~"),
      global_frame_id_("/map"),
      robot_frame_id_("/base_link"),
      octree_ptr_(0L),
      treat_unknown_as_free_(false),
      robot_height_(0.5)
{
    pnh_.param("global_frame_id", global_frame_id_, global_frame_id_);
    pnh_.param("robot_frame_id", robot_frame_id_, robot_frame_id_);
    pnh_.param("treat_unknown_as_free", treat_unknown_as_free_, treat_unknown_as_free_);
    pnh_.param("robot_height_", robot_height_, robot_height_);
    octree_sub_ = nh_.subscribe<octomap_msgs::Octomap>("octree_in", 1, &OctomapPathPlanner::onOctomap, this);
    goal_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("goal_in", 1, &OctomapPathPlanner::onGoal, this);
    ground_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("ground_cloud_out", 1, true);
    path_pub_ = nh_.advertise<nav_msgs::Path>("path_out", 1, true);
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


void OctomapPathPlanner::onGoal(const geometry_msgs::PoseStamped::ConstPtr& msg)
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
        tf_listener_.transformPose(global_frame_id_, robot_pose_local, robot_pose_);
        tf_listener_.transformPose(global_frame_id_, *msg, target_pose_);
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

    ROS_INFO("begin expanding octree");

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


void OctomapPathPlanner::computeGround()
{
    if(!octree_ptr_) return;

    ROS_INFO("begin computing ground");

    ground_pcl_.clear();

    for(octomap::OcTree::leaf_iterator it = octree_ptr_->begin(); it != octree_ptr_->end(); ++it)
    {
        if(isGround(it.getKey()))
        {
            pcl::PointXYZI point;
            point.x = it.getX();
            point.y = it.getY();
            point.z = it.getZ();
            point.intensity = std::numeric_limits<float>::infinity();
            ground_pcl_.push_back(point);
        }
    }

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
}


void OctomapPathPlanner::computeDistanceTransform()
{
    ROS_INFO("begin computing distance transform");

    // make octree for fast search in ground pcl:
    double res = octree_ptr_->getResolution();
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> pcl_octree(res);
    pcl_octree.setInputCloud(ground_pcl_.makeShared());
    pcl_octree.addPointsFromInputCloud();

    // find goal index in ground pcl:
    std::vector<int> pointIdx;
    std::vector<float> pointDistSq;
    pcl::PointXYZI goal;
    goal.x = target_pose_.pose.position.x;
    goal.y = target_pose_.pose.position.y;
    goal.z = target_pose_.pose.position.z;
    if(pcl_octree.nearestKSearch(goal, 1, pointIdx, pointDistSq) < 1)
    {
        ROS_ERROR("unable to find goal in ground pcl");
        return;
    }

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
        pcl_octree.radiusSearch(ground_pcl_[i], 1.8 * res, pointIdx, pointDistSq);

        for(std::vector<int>::iterator it = pointIdx.begin(); it != pointIdx.end(); it++)
        {
            int j = *it;

            // intensity value are initially set to infinity.
            // if i is finite it means it has already been labeled.
            if(isfinite(ground_pcl_[j].intensity)) continue;

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
        imin = fmin(imin, it->intensity);
        imax = fmax(imax, it->intensity);
    }
    ROS_INFO("intensity bounds before normalization: <%f, %f>", imin, imax);
    float d = imax - imin;
    for(pcl::PointCloud<pcl::PointXYZI>::iterator it = ground_pcl_.begin(); it != ground_pcl_.end(); ++it)
    {
        it->intensity = (it->intensity - imin) / d;
    }

    publishGroundCloud();

    ROS_INFO("finished computing distance transform");
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "octomap_path_planner");

    OctomapPathPlanner p;
    ros::spin();

    return 0;
}

