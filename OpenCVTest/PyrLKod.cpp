#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>

#include "Tests.h"

using namespace cv;
using namespace std;

static int MAX_FEATURES = 500;
static vector<KeyPoint> kpts1, kpts2;
static std::vector<DMatch> matches;
static Mat desc1, desc2;
static int picture_taken = 0;
//BFMatcher matcher(NORM_HAMMING);
static int initPoints;
static int started = 0;
static int rdpoints = 0;
// camera matrix for first camera at 0,0,0
static Mat K;
static vector<Point2f> points1, points2, init, reprojected;
// first 3d triangulated points
static vector<Point3f> init3dpoints, c3dpoints;
static vector<Mat> keyframes;

static TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 200, 0.03);
static Size subPixWinSize(10, 10);
static Size winSize(15, 15); // 31

static int frames = 0;
static Mat img1;
static Mat grey;
static Mat frame, img_matches;
static Mat temp, prevGray;
static Mat triOut;
static Mat ctriout;
static Mat sH = Mat::eye(3, 3, CV_64F);
static Mat M0 = Mat::eye(3, 4, CV_64F);
static Mat M1;

static Mat poseplot = Mat::zeros(480, 640, CV_8UC3);
static Mat totalT = Mat::zeros(3, 1, CV_64F);
// current rotation
static Mat R;
// current translation
static Mat T;
// current camera matrix
static Mat cM;
// current base cm all keyframe multiplied
static Mat ckM;

static Mat W;
static Mat mask;
static VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

static vector<uchar> status;
static vector<float> err;

void convertFromHom(Mat tri, vector<Point3f> *points3d)
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

void triangulate_points(Mat CM0, Mat CM1, vector<Point2f> poi1, vector<Point2f> poi2, vector<Point3f> *points3d)
{
	triangulatePoints(CM0, CM1, poi1, poi2, triOut);
	printf("row: %d, col: %d\n", triOut.rows, triOut.cols);
	convertFromHom(triOut, points3d);
}

void loop() {
	stream1.read(frame);
	//resize(frame, frame, Size(frame.cols*0.8, frame.rows*0.8));
	cvtColor(frame, grey, COLOR_BGR2GRAY);

	// move to function?
	if (!points1.empty()) {
		calcOpticalFlowPyrLK(prevGray, grey, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
		// remove bad tracks
		size_t k;
		for (size_t i = k = 0; i < points2.size(); i++) {
			if (!status[i])
				continue;
			points2[k] = points2[i];
			points1[k] = points1[i];
			init[k] = init[i];

			if(rdpoints){
				init3dpoints[k] = init3dpoints[i];
			}
			k++;
			circle(frame, points2[i], 2, Scalar(0, 255, 0), -1, 8);
			if (!rdpoints){
				line(frame, init[i], points2[i], Scalar(0, 255, 0));
			}
		}
		points1.resize(k);
		points2.resize(k);
		init.resize(k);
		if (rdpoints) {
			init3dpoints.resize(k);
		}
	}

	if (points1.size() > 8) {
		totalT = totalT + T;

		cv::Mat rvec(3, 1, cv::DataType<double>::type);
		cv::Mat tvec(3, 1, cv::DataType<double>::type);
		float scale = 0;
		if (init3dpoints.size() > 0) {
			solvePnPRansac(init3dpoints, points2, K, noArray(), rvec, tvec, false, 200,4);
			frames++;
			T = T + tvec;
			if (frames == 3) {
				T = T / 3;
				circle(poseplot, Point(200 + T.at<double>(0, 0) * 100, 200 + T.at<double>(1, 0) * 100), 2, Scalar(0, 255, 0));
				T = Mat::zeros(3, 1, CV_64F);
				frames = 0;
			}
		}
	}

	imshow("cam", frame);

	int key = waitKey(15);

	if (key == ' ') {
		if (started && !rdpoints) {
			rdpoints = 1;
			float f = K.at<double>(0, 0);
			Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
			Mat E = findEssentialMat(init, points2, f, pp, RANSAC, 0.99, 1.0, mask);
			int inliers = recoverPose(E, init, points2, R, T, f, pp);
			hconcat(R, T, M1);
			triangulate_points(K*M0, K*M1, init, points2, &init3dpoints);

		}
	}

	std::swap(points2, points1);
	cv::swap(prevGray, grey);

	if (key == ' ' && !rdpoints) {
		started = 1;
		// features and keypoints for object
		img1 = grey.clone();
		keyframes.push_back(img1);
		kpts1.clear();
		init.clear();
		goodFeaturesToTrack(img1, points1, MAX_FEATURES, 0.01, 15, Mat(), 3, 0, 0.04);
		//cornerSubPix(img1, points1, subPixWinSize, Size(-1, -1), termcrit);
		for (size_t i = 0; i < points1.size(); i++) {
			kpts1.push_back(cv::KeyPoint(points1[i], 1.f));
			init.push_back(Point2f(points1[i]));
		}
	}
	else if (key == 'q') {
		return;
	}
	imshow("pose", poseplot);
}

int PyrLKod() {
	//TODO move to function readcameramatrix
	

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

	while (true) {
		loop();
	}
	return 0;
}
