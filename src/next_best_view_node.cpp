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
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>

#include <octomap/octomap.h>
#include <octomap_ros/conversions.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include <pcl_conversions/pcl_conversions.h>

class NextBestView
{
public:
    NextBestView();
    ~NextBestView();
    bool isNearVoid(const octomap::point3d& p1, const unsigned char depth, const double res);
    void computeNextBestViews();
    void onOctomap(const octomap_msgs::Octomap::ConstPtr& m);
protected:
    std::string frame_id_;
    std::string robot_frame_id_;
    int num_clusters_;
    int min_computation_interval_;
    double normal_search_radius_;
    int min_pts_per_cluster_;
    double eps_angle_;
    double tolerance_;
    double boundary_angle_threshold_;
    ros::NodeHandle nh_;
    ros::NodeHandle private_node_handle_;
    octomap::OcTree *octree_ptr_;
    ros::Time last_computation_time_;
    ros::Publisher void_frontier_pub_;
    ros::Publisher posearray_pub_;
    std::vector<ros::Publisher> cluster_pub_;
    ros::Subscriber octree_sub_;
};


NextBestView::NextBestView() :
    frame_id_("/map"),
    robot_frame_id_("/base_link"),
    num_clusters_(3),
    min_computation_interval_(5 /*seconds*/),
    normal_search_radius_(0.4),
    min_pts_per_cluster_(5),
    eps_angle_(0.25),
    tolerance_(0.3),
    boundary_angle_threshold_(2.5),
    private_node_handle_("~"),
    octree_ptr_(0L)
{
    private_node_handle_.param("frame_id", frame_id_, frame_id_);
    private_node_handle_.param("robot_frame_id", robot_frame_id_, robot_frame_id_);
    private_node_handle_.param("num_clusters", num_clusters_, num_clusters_);
    private_node_handle_.param("min_computation_interval", min_computation_interval_, min_computation_interval_);
    private_node_handle_.param("normal_search_radius", normal_search_radius_, normal_search_radius_);
    private_node_handle_.param("min_pts_per_cluster", min_pts_per_cluster_, min_pts_per_cluster_);
    private_node_handle_.param("eps_angle", eps_angle_, eps_angle_);
    private_node_handle_.param("tolerance", tolerance_, tolerance_);
    private_node_handle_.param("boundary_angle_threshold", boundary_angle_threshold_, boundary_angle_threshold_);

    octree_sub_ = nh_.subscribe<octomap_msgs::Octomap>("octree_in", 1, &NextBestView::onOctomap, this);
    void_frontier_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("void_frontier", 1, false);
    posearray_pub_ = nh_.advertise<geometry_msgs::PoseArray>("poses", 1, false);
    for(int i = 0; i < num_clusters_; i++)
    {
        std::stringstream ss; ss << "cluster_pcl_" << (i+1);
        cluster_pub_.push_back(nh_.advertise<sensor_msgs::PointCloud2>(ss.str(), 1, false));
    }
}


NextBestView::~NextBestView()
{
    if(octree_ptr_) delete octree_ptr_;
}


/**
 * Check if a point is near the unknown space (by 1 voxel).
 * Note: this assumes the point is contained by a cell in the octree.
 */
bool NextBestView::isNearVoid(const octomap::point3d& p1, const unsigned char depth, const double res)
{
    for(int dz = 0; dz <= 0; dz += 2)
    {
        for(int dy = -1; dy <= 1; dy += 2)
        {
            for(int dx = -1; dx <= 1; dx += 2)
            {
                octomap::OcTreeNode *pNode = octree_ptr_->search(p1.x() + res * dx, p1.y() + res * dy, p1.z() + res * dz, depth);
                if(!pNode) return true;
            }
        }
    }
    return false;
}


static bool compareClusters(pcl::PointIndices c1, pcl::PointIndices c2)
{
    return (c1.indices.size() < c2.indices.size());
}


/**
 * Compute void frontier points (leaf points in free space adjacent to unknown space)
 */
void NextBestView::computeNextBestViews()
{
    const unsigned char depth = 16;
    const double res = octree_ptr_->getResolution();

    // compute void frontier:
    octomap::point3d_list pl;
    for(octomap::OcTree::leaf_iterator it = octree_ptr_->begin_leafs(depth); it != octree_ptr_->end_leafs(); it++)
    {
        if(octree_ptr_->isNodeOccupied(*it))
            continue;

        if(isNearVoid(it.getCoordinate(), depth, res))
            pl.push_back(it.getCoordinate());
    }
    if(!pl.size())
    {
        ROS_ERROR("Found no frontier points at depth %d!", depth);
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> border_pcl;
    border_pcl.resize(pl.size());
    border_pcl.header.frame_id = frame_id_;
    size_t i = 0;
    for(octomap::point3d_list::iterator it = pl.begin(); it != pl.end(); ++it)
    {
        border_pcl[i].x = it->x();
        border_pcl[i].y = it->y();
        border_pcl[i].z = it->z();
        i++;
    }

    sensor_msgs::PointCloud2 void_frontier_msg;
    pcl::toROSMsg(border_pcl, void_frontier_msg);
    void_frontier_pub_.publish(void_frontier_msg);

    // estimate normals:
    pcl::PointCloud<pcl::Normal> border_normals;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_estim;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree = boost::make_shared<pcl::search::KdTree<pcl::PointXYZ> > ();
    tree->setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> > (border_pcl));
    norm_estim.setSearchMethod(tree);
    norm_estim.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> > (border_pcl));
    norm_estim.setRadiusSearch(normal_search_radius_);
    norm_estim.compute(border_normals);

    // filter NaNs:
    pcl::PointIndices nan_indices;
    for (unsigned int i = 0; i < border_normals.points.size(); i++) {
        if (isnan(border_normals.points[i].normal[0]))
            nan_indices.indices.push_back(i);
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(border_pcl));
    extract.setIndices(boost::make_shared<const pcl::PointIndices>(nan_indices));
    extract.setNegative(true);
    extract.filter(border_pcl);
    pcl::ExtractIndices<pcl::Normal> nextract;
    nextract.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::Normal> >(border_normals));
    nextract.setIndices(boost::make_shared<const pcl::PointIndices>(nan_indices));
    nextract.setNegative(true);
    nextract.filter(border_normals);

    // tree object used for search
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree2 = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ> >();
    tree2->setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(border_pcl));
    // Decompose a region of space into clusters based on the euclidean distance between points, and the normal
    std::vector<pcl::PointIndices> clusters;
    pcl::extractEuclideanClusters<pcl::PointXYZ, pcl::Normal>(border_pcl, border_normals, tolerance_, tree2, clusters, eps_angle_, min_pts_per_cluster_);

    if(clusters.size() > 0)
    {
        std::sort(clusters.begin(), clusters.end(), compareClusters);
        pcl::PointCloud<pcl::PointXYZ> cluster_clouds[num_clusters_];

        geometry_msgs::PoseArray nbv_pose_array;

        for(unsigned int nc = 0; nc < clusters.size(); nc++) {
            if(nc == num_clusters_)
                break;

            // extract a cluster:
            pcl::PointCloud<pcl::PointXYZ> cluster_pcl;
            cluster_pcl.header = border_pcl.header;
            extract.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(border_pcl));
            extract.setIndices(boost::make_shared<const pcl::PointIndices>(clusters.back()));
            extract.setNegative(false);
            extract.filter(cluster_pcl);
            // extract normals of cluster:
            pcl::PointCloud<pcl::Normal> cluster_normals;
            cluster_normals.header = border_pcl.header;
            nextract.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::Normal> >(border_normals));
            nextract.setIndices(boost::make_shared<const pcl::PointIndices>(clusters.back()));
            nextract.setNegative(false);
            nextract.filter(cluster_normals);
            // find boundary points of cluster:
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree3 = boost::make_shared<pcl::search::KdTree<pcl::PointXYZ> >();
            tree3->setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(cluster_pcl));
            pcl::PointCloud<pcl::Boundary> boundary_pcl;
            pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> be;
            be.setSearchMethod(tree3);
            be.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(cluster_pcl));
            be.setInputNormals(boost::make_shared<pcl::PointCloud<pcl::Normal> >(cluster_normals));
            be.setRadiusSearch(0.5);
            be.setAngleThreshold(boundary_angle_threshold_);
            be.compute(boundary_pcl);

            geometry_msgs::Pose nbv_pose;
            for (unsigned int i = 0; i < boundary_pcl.points.size(); ++i) {
                if (boundary_pcl.points[i].boundary_point) {
                    nbv_pose.position.x = cluster_pcl.points[i].x;
                    nbv_pose.position.y = cluster_pcl.points[i].y;
                    nbv_pose.position.z = cluster_pcl.points[i].z;
                    tf::Vector3 axis(0, -cluster_normals.points[i].normal[2],
                            cluster_normals.points[i].normal[1]);
                    tf::Quaternion quat(axis, axis.length());
                    geometry_msgs::Quaternion quat_msg;
                    tf::quaternionTFToMsg(quat, quat_msg);
                    nbv_pose.orientation = quat_msg;
                    nbv_pose_array.poses.push_back(nbv_pose);
                }
            }

            cluster_clouds[nc] = cluster_pcl;

            // pop the just used cluster from indices:
            clusters.pop_back();
        }

        // visualize pose array:
        nbv_pose_array.header.frame_id = border_pcl.header.frame_id;
        nbv_pose_array.header.stamp = ros::Time::now();
        posearray_pub_.publish(nbv_pose_array);

        // visualize cluster pcls:
        for(int i = 0; i < num_clusters_; i++)
        {
            sensor_msgs::PointCloud2 cluster_pcl_msg;
            pcl::toROSMsg(cluster_clouds[i], cluster_pcl_msg);
            cluster_pub_[i].publish(cluster_pcl_msg);
        }
    }
}


/**
 * Octomap callback.
 *
 * It will skip if trying to compute poses more frequently than min_goal_interval.
 */
void NextBestView::onOctomap(const octomap_msgs::Octomap::ConstPtr& map)
{
    if((last_computation_time_ + ros::Duration(min_computation_interval_, 0)) > ros::Time::now())
        return;

    if(octree_ptr_) delete octree_ptr_;
    octree_ptr_ = octomap_msgs::binaryMsgToMap(*map);

    last_computation_time_ = ros::Time::now();
    computeNextBestViews();
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "next_best_view_node");

    NextBestView nbv;
    ros::spin();

    return 0;
}
