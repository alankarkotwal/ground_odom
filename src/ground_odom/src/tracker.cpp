/************************************************************************\
* ground_odom tracker							 *
* Stereo odometry from a downward-facing stereo camera on a vehicle	 *
* Alankar Kotwal <alankarkotwal13@gmail.com>				 *
* Created June 8, 2015 || Edited July 3, 2015				 *
\************************************************************************/

/*
	Include files: ROS headers
*/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float64.h>

/*
	Vision Libraries
*/
#include <opencv2/opencv.hpp>
#include <visp/vpImage.h>
#include <visp/vpImageConvert.h>
#include <visp/vpTemplateTrackerSSDInverseCompositional.h>
#include <visp/vpTemplateTrackerZNCCInverseCompositional.h>
#include <visp/vpTemplateTrackerWarpSRT.h>
#include <visp/vpTemplateTrackerWarpTranslation.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

/*
	C++ Libraries
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

/************************************************************************/

/*
	Macros
*/
#define PI 3.14159265
#define rad2deg(x) x*180/PI
#define deg2rad(x) x*PI/180

/************************************************************************/

/*
	Global variables
*/
cv::Mat left_image, right_image, left_image_init, right_image_init;
vpImage<unsigned char> left_visp, right_visp, left_visp_init, right_visp_init;
bool have_left_image = false, have_right_image = false, first = true;
ros::Publisher x_pub, y_pub, stereo_z_pub, tracker_z_pub, yaw_pub, pitch_pub, roll_pub;

/*
	Config parameters
*/
bool find_pitch_roll, show_images;
double image_resize_fraction, focal_length, baseline, pix_res, depth_crop_fraction, depth_overlap, depth_step_fraction, sac_threshold, position_crop_fraction, position_overlap, position_step_fraction;

/*
	ViSP stuff
*/
vpTemplateTrackerWarpTranslation depthWarp;
vpTemplateTrackerSSDInverseCompositional depthTracker(&depthWarp);
vpTemplateTrackerWarpSRT odomWarp;
vpTemplateTrackerZNCCInverseCompositional odomTracker(&odomWarp);
std::vector<vpImagePoint> v_ip;

/*
	Depth stuff
*/
int depth_start_x, depth_start_y, depth_n_x, depth_n_y, depth_step_x, depth_step_y, depth_block_width, depth_block_height;

/*
	Position stuff, do this
*/
//int depth_start_x, depth_start_y, depth_n_x, depth_n_y, depth_step_x, depth_step_y;

/*
	PCL and depth-RANSAC stuff
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::SACSegmentation<pcl::PointXYZ> seg;
std::vector<float> depths;

/*
	Odometry parameters
*/
float depth, depth_init, camera_x = 0, camera_y = 0, yaw = 0, pitch = 0, roll = 0;

/************************************************************************/

/*
	Functions
*/

float rejectOutliers(std::vector<float> depths) {

	/*
		Get median, as of now
	*/
	std::sort(depths.begin(), depths.end());
	int s = depths.size();
	if(s % 2) {
		return (depths[(s/2)-1] + depths[s/2])/2;
	}
	else {
		return depths[(s-1)/2];
	}		
}

void updateDepth() {

	if(first) {
	
		/*
			Calculate some stuff for a peaceful non-messy future
		*/
		depth_start_x = (1-depth_crop_fraction)*left_image.cols/2;
		depth_start_y = (1-depth_crop_fraction)*left_image.rows/2;
	
		int val_pix_x = depth_crop_fraction*left_image.cols;
		int val_pix_y = depth_crop_fraction*left_image.rows;
		depth_block_width = depth_step_fraction*val_pix_x;
		depth_block_height = depth_step_fraction*val_pix_y;
	
		depth_step_x = (1-depth_overlap)*depth_block_width;
		depth_step_y = (1-depth_overlap)*depth_block_height;
	
		depth_n_x = val_pix_x/depth_step_x;
		depth_n_y = val_pix_y/depth_step_y;
	}

	for(int i=0; i<depth_n_x; i++) {

		for(int j=0; j<depth_n_y; j++) {

			/*
				Track the relevent patch
			*/
			v_ip.clear();
			vpImagePoint ip;
		
			ip.set_ij(depth_start_x+i*depth_step_x, depth_start_y+j*depth_step_y); v_ip.push_back(ip); // Top left
			ip.set_ij(depth_start_x+i*depth_step_x+depth_block_width, depth_start_y+j*depth_step_y); v_ip.push_back(ip); // Top right
			ip.set_ij(depth_start_x+i*depth_step_x, depth_start_y+j*depth_step_y+depth_block_height); v_ip.push_back(ip);
			ip.set_ij(depth_start_x+i*depth_step_x+depth_block_width, depth_start_y+j*depth_step_y); v_ip.push_back(ip);
			ip.set_ij(depth_start_x+i*depth_step_x, depth_start_y+j*depth_step_y+depth_block_height); v_ip.push_back(ip);
			ip.set_ij(depth_start_x+i*depth_step_x+depth_block_width, depth_start_y+j*depth_step_y+depth_block_height); v_ip.push_back(ip);

			depthTracker.resetTracker();

			try {
				depthTracker.initFromPoints(left_visp, v_ip, false);
				depthTracker.track(right_visp);

				vpColVector p = depthTracker.getp();
				float dep = std::abs(focal_length*baseline*image_resize_fraction/(pix_res*p[0]));
			
				if(find_pitch_roll) {
					pcl::PointXYZ pt;
					pt.x = dep*pix_res/focal_length*(depth_start_x+i*depth_step_x+depth_block_width/2-left_image.cols/2);
					pt.y = dep*pix_res/focal_length*(depth_start_y+j*depth_step_y+depth_block_height/2-left_image.rows/2);
					pt.z = dep;
			
					cloud->push_back(pt);
				}
				
				depths.push_back(dep);
			}
			catch(vpException &e) {
		
				//ROS_WARN("Couldn't track plane!");
			}
		}
	}
	
	if(find_pitch_roll) {

		/*
			Fit plane with RANSAC
		*/
		pcl::ModelCoefficients coefficients;
		pcl::PointIndices inliers;
		seg.setInputCloud(cloud);
		seg.segment(inliers, coefficients);
		cloud->clear();
		//ROS_INFO("%f %f %f %f", coefficients.values[0], coefficients.values[1], coefficients.values[2], coefficients.values[3]);
		//depth = coefficients.values[3]/coefficients.values[2];
	}
	
	depth = rejectOutliers(depths);
	
	//ROS_INFO("%f", depth);
	depths.clear();
}

void updateOdometry() {

	/*
		Track the center patch and get parameters
	*/
	v_ip.clear();
	vpImagePoint ip;
	ip.set_ij(left_image.cols/4, left_image.rows/4); v_ip.push_back(ip);
	ip.set_ij(3*left_image.cols/4, left_image.rows/4); v_ip.push_back(ip);
	ip.set_ij(left_image.cols/4, 3*left_image.rows/4); v_ip.push_back(ip);
	ip.set_ij(3*left_image.cols/4, left_image.rows/4); v_ip.push_back(ip);
	ip.set_ij(left_image.cols/4, 3*left_image.rows/4); v_ip.push_back(ip);
	ip.set_ij(3*left_image.cols/4, 3*left_image.rows/4); v_ip.push_back(ip);

	odomTracker.resetTracker();

	odomTracker.initFromPoints(left_visp_init, v_ip, false);
	odomTracker.track(left_visp);

	vpColVector p = odomTracker.getp();
	
	/*
		Get average yaw for x and y updates, and update yaw
	*/
	float avgYaw = (deg2rad(yaw) + p[1])/2;
	yaw += rad2deg(p[1]);

	/*
		Account for scale
	*/
	resize(left_image_init, left_image_init, cv::Size(0, 0), depth_init/depth, depth_init/depth, cv::INTER_LINEAR);

	/*
		Construct the ROI and the template from the previous left image
	*/
	cv::Rect window((float)left_image_init.cols/4, (float)left_image_init.rows/4,
					(float)left_image_init.cols/2, (float)left_image_init.rows/2);
	cv::Mat temp_image = left_image_init(window);	

	/*
		Match the template to the left using normalised cross-correlation
	*/
	cv::Mat corrs;
	cv::matchTemplate(left_image, temp_image, corrs, CV_TM_CCOEFF_NORMED);

	/*
		Get the maximum and hence the odometry
	*/
	cv::Point max;
	cv::minMaxLoc(corrs, NULL, NULL, NULL, &max, cv::Mat());

	double disp_x, disp_y;
	disp_x = max.x - corrs.cols/2;
	disp_y = max.y - corrs.rows/2;
	//ROS_INFO("%f %f", disp_x, disp_y);
	/*disp_x = p[2];
	disp_y = p[3];*/
	camera_x += depth*pix_res*(disp_x*std::cos(avgYaw) + disp_y*std::sin(avgYaw))/(image_resize_fraction*focal_length);
	camera_y += depth*pix_res*(disp_y*std::cos(avgYaw) - disp_x*std::sin(avgYaw))/(image_resize_fraction*focal_length);

	ROS_INFO("My co-ordinates are (%f, %f, %f) cm, heading %f degrees.", 100*camera_x, 100*camera_y, 100*depth, yaw);

	/*
		Publish my findings
	*/
	std_msgs::Float64 msg;
	msg.data = camera_x;
	x_pub.publish(msg);
	msg.data = camera_y;
	y_pub.publish(msg);
	msg.data = depth;
	stereo_z_pub.publish(msg);
	msg.data = depth_init/p[0]; // Check this
	tracker_z_pub.publish(msg);
	msg.data = yaw;
	yaw_pub.publish(msg);
	if(find_pitch_roll) {
		msg.data = pitch;
		pitch_pub.publish(msg);
		msg.data = roll;
		roll_pub.publish(msg);
	}
}

/************************************************************************/

// Callbacks

// Left image callback
void leftImageCb(const sensor_msgs::ImageConstPtr& msg) {

	/*
		Get image
	*/
	cv_bridge::CvImagePtr cv_ptr;

	try {
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch(cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	left_image = cv_ptr->image;

	/*
		Pre-process
	*/
	cv::resize(left_image, left_image, cv::Size(0, 0), image_resize_fraction, image_resize_fraction, cv::INTER_LINEAR);
	cv::cvtColor(left_image, left_image, CV_BGR2GRAY);

	if(show_images) {

		/*
			Show images
		*/
		cv::imshow("Left", left_image);
		cv::waitKey(1);
	}

	/*
		Convert to ViSP image
	*/
	vpImageConvert::convert(left_image, left_visp);

	/*
		Approximate-synchronisation
	*/
	if(have_right_image) {
		updateDepth();
		if(!first) {
			updateOdometry();
		} else {
			first = false;
		}
		left_image_init = left_image;
		right_image_init = right_image;
		left_visp_init = left_visp;
		right_visp_init = right_visp;
		depth_init = depth;
		have_right_image = false;
		have_left_image = false;
	}
	else {
		have_left_image = true;
	}
}

// Right image callback
void rightImageCb(const sensor_msgs::ImageConstPtr& msg) {

	/*
		Get image
	*/
	cv_bridge::CvImagePtr cv_ptr;

	try {
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	right_image = cv_ptr->image;

	/*
		Pre-process
	*/
	cv::resize(right_image, right_image, cv::Size(0, 0), image_resize_fraction, image_resize_fraction, cv::INTER_LINEAR);
	cv::cvtColor(right_image, right_image, CV_BGR2GRAY);

	if(show_images) {

		/*
			Show images
		*/
		cv::imshow("Right", right_image);
		cv::waitKey(1);
	}

	/*
		Convert to ViSP image
	*/
	vpImageConvert::convert(right_image, right_visp);

	/*
		Approximate-synchronisation
	*/
	if(have_left_image) {
		updateDepth();
		if(!first) {
			updateOdometry();
		} else {
			first = false;
		}
		left_image_init = left_image;
		right_image_init = right_image;
		left_visp_init = left_visp;
		right_visp_init = right_visp;
		depth_init = depth;
		have_right_image = false;
		have_left_image = false;
	}
	else {
		have_right_image = true;
	}

}

/************************************************************************/

// Main

int main(int argc, char** argv) {
	
	/*
		Set the node up
	*/
	ros::init(argc, argv, "tracker");
	ros::NodeHandle nh;
	
	/*
		Get parameters
	*/
	nh.getParam("/tracker/run_config/find_pitch_roll", find_pitch_roll);
	nh.getParam("/tracker/camera_params/resize_factor", image_resize_fraction);
	// These should probably come from the camera_info message
	nh.getParam("/tracker/camera_params/focal_length", focal_length);
	nh.getParam("/tracker/camera_params/stereo_baseline", baseline);
	nh.getParam("/tracker/camera_params/pix_res", pix_res);
	nh.getParam("/tracker/depth_params/crop_fraction", depth_crop_fraction);
	nh.getParam("/tracker/depth_params/overlap", depth_overlap);
	nh.getParam("/tracker/depth_params/step_fraction", depth_step_fraction);
	nh.getParam("/tracker/depth_params/sac_threshold", sac_threshold);
	nh.getParam("/tracker/position_params/crop_fraction", position_crop_fraction);
	nh.getParam("/tracker/position_params/overlap", position_overlap);
	nh.getParam("/tracker/position_params/step_fraction", position_step_fraction);
	nh.getParam("/tracker/display/show_images", show_images);	
	
	/*
		Subscribe to and advert topics
	*/
	image_transport::ImageTransport it(nh);
	image_transport::Subscriber left_image_sub, right_image_sub;
	left_image_sub = it.subscribe("/duo3d_camera/left/image_rect", 1, &leftImageCb);
	right_image_sub = it.subscribe("/duo3d_camera/right/image_rect", 1, &rightImageCb);
	
	x_pub = nh.advertise<std_msgs::Float64>("/tracker/x", 1000);
	y_pub = nh.advertise<std_msgs::Float64>("/tracker/y", 1000);
	stereo_z_pub = nh.advertise<std_msgs::Float64>("/tracker/stereo_z", 1000);
	tracker_z_pub = nh.advertise<std_msgs::Float64>("/tracker/tracker_z", 1000);
	yaw_pub = nh.advertise<std_msgs::Float64>("/tracker/yaw", 1000);
	pitch_pub = nh.advertise<std_msgs::Float64>("/tracker/pitch", 1000);
	roll_pub = nh.advertise<std_msgs::Float64>("/tracker/roll", 1000);
	
	/*
		Initialise our trackers and segmenter
	*/
	// Should these parameters come from the parameter server?
	depthTracker.setSampling(2, 2);
	depthTracker.setLambda(0.001);
	depthTracker.setThresholdGradient(60.);
	depthTracker.setIterationMax(800);
	depthTracker.setPyramidal(2, 1);
	
	odomTracker.setSampling(2, 2);
	odomTracker.setLambda(0.001);
	odomTracker.setThresholdGradient(60.);
	odomTracker.setIterationMax(800);
	odomTracker.setPyramidal(2, 1);
	
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(sac_threshold);
	
	/*
		ROS main thread
	*/
	ros::spin();

	return 0;
}
