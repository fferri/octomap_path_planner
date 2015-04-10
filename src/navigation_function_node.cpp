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

namespace pcl
{
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
}

class NavigationFunction
{
protected:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    std::string frame_id_;
    std::string robot_frame_id_;
    ros::Subscriber octree_sub_;
    ros::Subscriber goal_point_sub_;
    ros::Subscriber goal_pose_sub_;
    ros::Publisher ground_pub_;
    ros::Publisher obstacles_pub_;
    tf::TransformListener tf_listener_;    
    geometry_msgs::PoseStamped robot_pose_;
    geometry_msgs::PoseStamped goal_;
    octomap::OcTree* octree_ptr_;
    pcl::PointCloud<pcl::PointXYZI> ground_pcl_;
    pcl::PointCloud<pcl::PointXYZ> obstacles_pcl_;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::Ptr ground_octree_ptr_;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr obstacles_octree_ptr_;
    bool treat_unknown_as_free_;
    double robot_height_;
    double robot_radius_;
    double max_superable_height_;
    double ground_voxel_connectivity_;
public:
    NavigationFunction();
    ~NavigationFunction();
    void onOctomap(const octomap_msgs::Octomap::ConstPtr& msg);
    void onGoal(const geometry_msgs::PointStamped::ConstPtr& msg);
    void onGoal(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void expandOcTree();
    bool isGround(const octomap::OcTreeKey& key);
    bool isObstacle(const octomap::OcTreeKey& key);
    bool isNearObstacle(const pcl::PointXYZI& point);
    void filterInflatedRegionFromGround();
    void computeGround();
    void projectGoalPositionToGround();
    void publishGroundCloud();
    int getGoalIndex();
    void getNeighboringGroundPoints(int index, std::vector<int>& neighbors, double search_radius, bool exclude_already_labeled);
    void computeDistanceTransform();
    double getAverageIntensity(int index, double search_radius);
    void smoothIntensity(double search_radius);
    void normalizeIntensity();
};


NavigationFunction::NavigationFunction()
    : pnh_("~"),
      frame_id_("/map"),
      robot_frame_id_("/base_link"),
      octree_ptr_(0L),
      treat_unknown_as_free_(false),
      robot_height_(0.5),
      robot_radius_(0.5),
      max_superable_height_(0.2),
      ground_voxel_connectivity_(1.8)
{
    pnh_.param("frame_id", frame_id_, frame_id_);
    pnh_.param("robot_frame_id", robot_frame_id_, robot_frame_id_);
    pnh_.param("treat_unknown_as_free", treat_unknown_as_free_, treat_unknown_as_free_);
    pnh_.param("robot_height", robot_height_, robot_height_);
    pnh_.param("robot_radius", robot_radius_, robot_radius_);
    pnh_.param("max_superable_height", max_superable_height_, max_superable_height_);
    pnh_.param("ground_voxel_connectivity", ground_voxel_connectivity_, ground_voxel_connectivity_);
    octree_sub_ = nh_.subscribe<octomap_msgs::Octomap>("octree_in", 1, &NavigationFunction::onOctomap, this);
    goal_point_sub_ = nh_.subscribe<geometry_msgs::PointStamped>("goal_point_in", 1, &NavigationFunction::onGoal, this);
    goal_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("goal_pose_in", 1, &NavigationFunction::onGoal, this);
    ground_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("ground_cloud_out", 1, true);
    obstacles_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obstacles_cloud_out", 1, true);
    ground_pcl_.header.frame_id = frame_id_;
    obstacles_pcl_.header.frame_id = frame_id_;
}


NavigationFunction::~NavigationFunction()
{
    if(octree_ptr_) delete octree_ptr_;
}


void NavigationFunction::onOctomap(const octomap_msgs::Octomap::ConstPtr& msg)
{
    if(octree_ptr_) delete octree_ptr_;
    octree_ptr_ = octomap_msgs::binaryMsgToMap(*msg);

    expandOcTree();
    computeGround();
    computeDistanceTransform();
}


void NavigationFunction::onGoal(const geometry_msgs::PointStamped::ConstPtr& msg)
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
        projectGoalPositionToGround();
        ROS_INFO("goal set to point (%f, %f, %f)",
            goal_.pose.position.x, goal_.pose.position.y, goal_.pose.position.z);
    }
    catch(tf::TransformException& ex)
    {
        ROS_ERROR("Failed to lookup robot position: %s", ex.what());
    }

    computeDistanceTransform();
}


void NavigationFunction::onGoal(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    try
    {
        tf_listener_.transformPose(frame_id_, *msg, goal_);
        projectGoalPositionToGround();
        ROS_INFO("goal set to pose (%f, %f, %f), (%f, %f, %f, %f)",
                goal_.pose.position.x, goal_.pose.position.y, goal_.pose.position.z,
                goal_.pose.orientation.x, goal_.pose.orientation.y, goal_.pose.orientation.z,
                goal_.pose.orientation.w);
    }
    catch(tf::TransformException& ex)
    {
        ROS_ERROR("Failed to lookup robot position: %s", ex.what());
    }

    computeDistanceTransform();
}


void NavigationFunction::expandOcTree()
{
    if(!octree_ptr_) return;

    unsigned int maxDepth = octree_ptr_->getTreeDepth();

    // expand collapsed occupied nodes until all occupied leaves are at maximum depth
    std::vector<octomap::OcTreeNode*> collapsed_occ_nodes;

    size_t initial_size = octree_ptr_->size();
    size_t num_rounds = 0;
    size_t expanded_nodes = 0;

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

        expanded_nodes += collapsed_occ_nodes.size();
        num_rounds++;
    } while(collapsed_occ_nodes.size() > 0);

    //ROS_INFO("received octree of %ld nodes; expanded %ld nodes in %ld rounds.", initial_size, expanded_nodes, num_rounds);
}


bool NavigationFunction::isGround(const octomap::OcTreeKey& key)
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


bool NavigationFunction::isObstacle(const octomap::OcTreeKey& key)
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


bool NavigationFunction::isNearObstacle(const pcl::PointXYZI& point)
{
    std::vector<int> pointIdx2;
    std::vector<float> pointDistSq2;
    pcl::PointXYZ p;
    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    int num_points = obstacles_octree_ptr_->nearestKSearch(p, 1, pointIdx2, pointDistSq2);
    return num_points >= 1 && pointDistSq2[0] < (robot_radius_ * robot_radius_);
}


void NavigationFunction::filterInflatedRegionFromGround()
{
    for(int i = 0; i < ground_pcl_.size(); )
    {
        if(isNearObstacle(ground_pcl_[i]))
        {
            std::swap(*(ground_pcl_.begin() + i), ground_pcl_.back());
            ground_pcl_.points.pop_back();
        }
        else i++;
    }
    ground_pcl_.width = ground_pcl_.points.size();
    ground_pcl_.height = 1;
}


void NavigationFunction::computeGround()
{
    if(!octree_ptr_) return;

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

    obstacles_octree_ptr_ = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(res));
    obstacles_octree_ptr_->setInputCloud(obstacles_pcl_.makeShared());
    obstacles_octree_ptr_->addPointsFromInputCloud();

    filterInflatedRegionFromGround();

    ground_octree_ptr_ = pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::Ptr(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>(res));
    ground_octree_ptr_->setInputCloud(ground_pcl_.makeShared());
    ground_octree_ptr_->addPointsFromInputCloud();
}


void NavigationFunction::projectGoalPositionToGround()
{
    pcl::PointXYZI goal;
    goal.x = goal_.pose.position.x;
    goal.y = goal_.pose.position.y;
    goal.z = goal_.pose.position.z;
    std::vector<int> pointIdx;
    std::vector<float> pointDistSq;
    if(ground_octree_ptr_->nearestKSearch(goal, 1, pointIdx, pointDistSq) < 1)
    {
        ROS_ERROR("Failed to project goal position to ground pcl");
        return;
    }
    int i = pointIdx[0];
    goal_.pose.position.x = ground_pcl_[i].x;
    goal_.pose.position.y = ground_pcl_[i].y;
    goal_.pose.position.z = ground_pcl_[i].z;
}


void NavigationFunction::publishGroundCloud()
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


int NavigationFunction::getGoalIndex()
{
    // find goal index in ground pcl:
    std::vector<int> pointIdx;
    std::vector<float> pointDistSq;
    pcl::PointXYZI goal;
    goal.x = goal_.pose.position.x;
    goal.y = goal_.pose.position.y;
    goal.z = goal_.pose.position.z;
    if(ground_octree_ptr_->nearestKSearch(goal, 1, pointIdx, pointDistSq) < 1)
    {
        return -1;
    }
    else
    {
        return pointIdx[0];
    }
}


void NavigationFunction::getNeighboringGroundPoints(int index, std::vector<int>& neighbors, double search_radius, bool exclude_already_labeled)
{
    neighbors.clear();
    std::vector<float> pointDistSq;
    ground_octree_ptr_->radiusSearch(ground_pcl_[index], search_radius, neighbors, pointDistSq);
}


void NavigationFunction::computeDistanceTransform()
{
    if(ground_pcl_.size() == 0)
    {
        ROS_INFO("skip computing distance transform because ground_pcl_ is empty");
        return;
    }

    // find goal index in ground pcl:
    int goal_idx = getGoalIndex();
    if(goal_idx == -1)
    {
        ROS_ERROR("unable to find goal in ground pcl");
        return;
    }

    double search_radius = ground_voxel_connectivity_ * octree_ptr_->getResolution();

    // distance to goal is zero (stored in the intensity channel):
    ground_pcl_[goal_idx].intensity = 0.0;

    std::queue<int> q;
    q.push(goal_idx);
    while(!q.empty())
    {
        int i = q.front();
        q.pop();

        std::vector<int> neighbors;
        getNeighboringGroundPoints(i, neighbors, search_radius, true);

        for(std::vector<int>::iterator it = neighbors.begin(); it != neighbors.end(); it++)
        {
            int j = *it;

            // intensity value are initially set to infinity.
            // if i is finite it means it has already been labeled.
            if(std::isfinite(ground_pcl_[j].intensity)) continue;

            // otherwise, label it:
            ground_pcl_[j].intensity = ground_pcl_[i].intensity +
                dist(ground_pcl_[j], ground_pcl_[i]);

            // continue exploring neighbours:
            q.push(j);
        }
    }

    //smoothIntensity(search_radius);
    normalizeIntensity();

    publishGroundCloud();
}


double NavigationFunction::getAverageIntensity(int index, double search_radius)
{
    std::vector<int> pointIdx;
    std::vector<float> pointDistSq;
    ground_octree_ptr_->radiusSearch(ground_pcl_[index], search_radius, pointIdx, pointDistSq);

    if(pointIdx.size() == 0) return std::numeric_limits<float>::infinity();

    double i = 0.0;
    for(std::vector<int>::iterator it = pointIdx.begin(); it != pointIdx.end(); ++it)
    {
        i += ground_pcl_[*it].intensity;
    }
    i /= (double)pointIdx.size();
    return i;
}


void NavigationFunction::smoothIntensity(double search_radius)
{
    std::vector<double> smoothed_intensity;
    smoothed_intensity.resize(ground_pcl_.size());
    for(size_t i = 0; i < ground_pcl_.size(); i++)
    {
        smoothed_intensity[i] = getAverageIntensity(i, search_radius);
    }
    for(size_t i = 0; i < ground_pcl_.size(); i++)
    {
        ground_pcl_[i] = smoothed_intensity[i];
    }
}


void NavigationFunction::normalizeIntensity()
{
    float imin = std::numeric_limits<float>::infinity();
    float imax = -std::numeric_limits<float>::infinity();
    for(pcl::PointCloud<pcl::PointXYZI>::iterator it = ground_pcl_.begin(); it != ground_pcl_.end(); ++it)
    {
        if(!std::isfinite(it->intensity)) continue;
        imin = fmin(imin, it->intensity);
        imax = fmax(imax, it->intensity);
    }
    const float eps = 0.01;
    float d = imax - imin + eps;
    for(pcl::PointCloud<pcl::PointXYZI>::iterator it = ground_pcl_.begin(); it != ground_pcl_.end(); ++it)
    {
        if(std::isfinite(it->intensity))
            it->intensity = (it->intensity - imin) / d;
        else
            it->intensity = 1.0;
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "navigation_function");

    NavigationFunction p;
    ros::spin();

    return 0;
}

