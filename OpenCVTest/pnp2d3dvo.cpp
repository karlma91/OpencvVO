#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;
#include <iostream>
#include <vector>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/viz.hpp>
#include "Tests.h"

static int MAX_FEATURES = 1000;
static vector<KeyPoint> kpts1, kpts2, kpts3, fkpts;
static std::vector<DMatch> matches;
std::vector< DMatch > good_matches;
std::vector<KeyPoint> matched1, matched2;
static Mat desc1, desc2, desc3, fdesc;

static int first_frame_taken = 0;
static int second_frame_taken = 0;
Mat first_frame, second_frame;
// camera matrix for first camera at 0,0,0
static Mat K;
static vector<Point2f> points1, points2;
// first 3d triangulated points
static vector<Point3f> init3dpoints, camposes, temp3d;

static vector<viz::Color> colors;

static Mat grey, prevGray;
static Mat frame;

static Mat sH = Mat::eye(3, 3, CV_64F);
static Mat M0 = Mat::eye(3, 4, CV_64F);
static Mat M1;

static Mat totalT = Mat::eye(4, 4, CV_64F);
// current rotation
static Mat R;
// current translation
static Mat T;
// current camera matrix

static Mat mask;
static Mat pointcloud(0, 0, CV_64FC3);
static Mat allColors(0, 0, CV_8UC3);

//static VideoCapture stream1("C:\\Users\\Karlmka\\Dropbox\\unik4690\\20150515_172609.mp4");   //0 is the id of video device.0 if you have only one camera.
static VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

static vector<uchar> status;
static vector<float> err;
// essential matrix
static Mat E;
//static Ptr<ORB> detector;
static Ptr<ORB> descriptor;
static Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
static BFMatcher matcher(NORM_HAMMING);

/// Create a window
static viz::Viz3d myWindow("VIZ");


static void convertFromHom(Mat tri, vector<Point3f> *points3d)
{
	points3d->clear();
	for (int i = 0; i < tri.cols; i++) {
		float s = tri.at<float>(3, i);
		float x = tri.at<float>(0, i) / s;
		float y = tri.at<float>(1, i) / s;
		float z = tri.at<float>(2, i) / s;
		points3d->push_back(Point3f(x, y, z));
	}
}

static void triangulate_points(Mat CM0, Mat CM1, vector<Point2f> poi1, vector<Point2f> poi2, vector<Point3f> *points3d)
{
	Mat triOut;
	triangulatePoints(CM0, CM1, poi1, poi2, triOut);
	convertFromHom(triOut, points3d);
}
static void draw_matches(Mat frame, vector<KeyPoint> kp1, vector<KeyPoint> kp2)
{
	for (int i = 0; i < kp1.size(); i++) {
		line(frame, kp1[i].pt, kp2[i].pt, Scalar(0, 255, 0));
	}
}

static void draw_key_points(Mat frame, vector<KeyPoint> kpts,int size, Scalar color) {
	for (int i = 0; i < kpts.size(); i++) {
		circle(frame, kpts[i].pt, size, color);
	}
}

static void take_first_frame(vector<KeyPoint> kp1, Mat desc1, Mat grey)
{
	first_frame_taken = 1;
	grey.copyTo(first_frame);
}

static void just_match(vector<KeyPoint> kp1, Mat des1, vector<KeyPoint> kp2, Mat des2, Mat grey) {

	if (des2.cols > 5) {
		matcher.match(des1, des2, matches);
		if (matches.size() > 5) {
			double max_dist = 0; double min_dist = 1000;
			matched1.clear();
			matched2.clear();
			good_matches.clear();
			for (int i = 0; i < matches.size(); i++) {
				if (matches[i].distance < 20) {
					good_matches.push_back(matches[i]);
					matched1.push_back(kp1[matches[i].queryIdx]);
					matched2.push_back(kp2[matches[i].trainIdx]);
					if (init3dpoints.size()>0) {
						temp3d.push_back(init3dpoints[matches[i].queryIdx]);
					}
				}
			}

			/*if (init3dpoints.size()>0){
				float f = K.at<double>(0, 0);
				Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
				vector<Point2f> tt1, tt2;
				KeyPoint::convert(matched1, tt1);
				KeyPoint::convert(matched2, tt2);
				E = findEssentialMat(tt1, tt2, f, pp, RANSAC, 0.995, 2.0, mask);
				int k = 0;
				for (int i = 0; i < mask.rows; i++){
					if ((int)mask.at<uchar>(i, 0)){
						matched1[k] = matched1[i];
						matched2[k] = matched2[i];
						temp3d[k] = temp3d[k];
						k++;
					}
				}
				matched1.resize(k);
				matched2.resize(k);
				temp3d.resize(k);
			}*/
		}
	}
}
static void take_second_frame_and_triangulate(vector<KeyPoint> kp1, Mat des1, vector<KeyPoint> kp2, Mat des2, Mat colors)
{
	if (good_matches.size() > 10) {
		grey.copyTo(second_frame);
		float f = K.at<double>(0, 0);
		Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
		KeyPoint::convert(matched1, points1);
		KeyPoint::convert(matched2, points2);
		E = findEssentialMat(points1, points2, f, pp, RANSAC, 0.995, 2.0, mask);
		int inliers = recoverPose(E, points1, points2, R, T, f, pp, mask);
		if (inliers > 10){
			hconcat(R, T, M1);
			camposes.push_back(Point3f(0, 0, 0));
			camposes.push_back(Point3f(T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0)));
			vector<Point2f> ny1, ny2;
			for (int i = 0; i < mask.rows; i++)
			{
				if ((int)mask.at<uchar>(i, 0) && (norm(points1[i] - points2[i]) > 3)){
					ny1.push_back(points1[i]);
					ny2.push_back(points2[i]);
					fkpts.push_back(matched2[i]);
				}
			}
			descriptor->compute(grey, fkpts, fdesc);
			triangulate_points(K*M0, K*M1, ny1, ny2, &init3dpoints);
			for (int i = 0; i < (int)init3dpoints.size(); i++) {
				Vec3b col = colors.at<Vec3b>(points2[i].y, points2[i].x);
				allColors.push_back(col);
			}
			second_frame_taken = 1;
			return;

		}
	}
	printf("Bad initialisation redo\n");
}

static void update()
{
	if (!first_frame_taken) {
		kpts1.clear();
		detector->detect(grey, kpts1);
		KeyPointsFilter::retainBest(kpts1, MAX_FEATURES);
		descriptor->compute(grey, kpts1, desc1);
	}
	else if (!second_frame_taken) {
		detector->detect(grey, kpts2);
		KeyPointsFilter::retainBest(kpts2, MAX_FEATURES);
		descriptor->compute(grey, kpts2, desc2);
		just_match(kpts1, desc1, kpts2, desc2, grey);
	}
	else{
		detector->detect(grey, kpts3);
		KeyPointsFilter::retainBest(kpts3, MAX_FEATURES);
		descriptor->compute(grey, kpts3, desc3);
		temp3d.clear();
		just_match(fkpts, fdesc, kpts3, desc3, grey);
		vector<Point2f> tt1, tt2;
		Mat rvec(3, 1, DataType<double>::type);
		Mat tvec(3, 1, DataType<double>::type);
		KeyPoint::convert(matched1, tt1);
		KeyPoint::convert(matched2, tt2);
		solvePnPRansac(temp3d, tt2, K, noArray(), rvec, tvec, false, 100, 8);
		T = tvec;
		Rodrigues(rvec, R);
	}
}

static void render()
{
	if (!first_frame_taken){
		draw_key_points(frame, kpts1,3,Scalar(255,0,0));
	}
	else if (!second_frame_taken) {
		draw_matches(frame, matched1, matched2);
	}
	else{
		draw_key_points(frame, fkpts, 3, Scalar(0, 0, 255));
		draw_matches(frame, matched1, matched2);
	}
}
static vector<Point2f> ny1, ny2;
static void loop() {
	stream1.read(frame);
	cvtColor(frame, grey, COLOR_BGR2GRAY);

	update();
	render();

	imshow("cam", frame);

	int key = waitKey(15);
	if (key == ' ') {
		if (!first_frame_taken) {
			take_first_frame(kpts1,desc1,grey);
		}
		else{
			take_second_frame_and_triangulate(kpts1, desc1, kpts2, desc2, frame);
		}
	}
	else if (key == 'q') {
		return;
	}

	if (init3dpoints.size() > 0) {
		viz::WCloud cw(camposes);
		Mat testt(init3dpoints);
		viz::WCloud cws(testt, allColors);
		cw.setRenderingProperty(viz::POINT_SIZE, 6);
		cws.setRenderingProperty(viz::POINT_SIZE, 3);
		myWindow.showWidget("CloudWidget1", cw);
		myWindow.showWidget("CloudWidget2", cws);

		Mat fD = (Mat_<double>(3, 1) << 0, 0, 1);
		Mat tmp = T + fD;// R*fD;
		Vec3d cam_pos(T), cam_focal_point(tmp), cam_y_dir(0.0f, -1.0f, 0.0f);
		/// We can get the pose of the cam using makeCameraPose
		Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

		viz::WCameraPosition cpw(0.5); // Coordinate axes
		viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
		myWindow.showWidget("CPW", cpw, cam_pose);
		myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);

	}
	myWindow.spinOnce(1, true);
}


int pnp2d3d() {
	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}
	// read camera matrix
	FileStorage fs("C:/Users/Karlmka/Dropbox/unik4690/Kamerakalibrering/camera.yml", FileStorage::READ);
	fs["camera_matrix"] >> K;
	fs.release();
	printf("K matrix: \n");
	print(K);
	printf("\n");

	//detector = ORB::create();
	//detector->setMaxFeatures(1000);
	descriptor = ORB::create();

	/// Add coordinate axes
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());;

	while (true) {
		loop();
	}
	return 0;
}
