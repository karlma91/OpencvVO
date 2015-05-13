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
static vector<Point3f> init3dpoints;
static vector<Mat> keyframes;

static TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 200, 0.03);
static Size subPixWinSize(10, 10);
static Size winSize(15, 15); // 31

static Mat img1;
static Mat grey;
static Mat frame, img_matches;
static Mat temp, prevGray;
static Mat triOut;
static Mat ctriout;
static Mat sH = Mat::eye(3, 3, CV_64F);
static Mat M0 = Mat::eye(3, 4, CV_64F);
static Mat poseplot = Mat::zeros(480, 640, CV_8UC3);
static Mat totalT = Mat::zeros(3, 1, CV_64F);
static Mat R;
static Mat T;
static Mat W;
static Mat mask;
static VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

static vector<uchar> status;
static vector<float> err;

void convertFromHom(Mat tri, vector<Point3f> *points3d)
{
	points3d->clear();
	for (int i = 0; i < triOut.cols; i++) {
		float s = triOut.at<float>(3, i);
		float x = triOut.at<float>(0, i) / s;
		float y = triOut.at<float>(1, i) / s;
		float z = triOut.at<float>(2, i) / s;
		points3d->push_back(Point3f(x, y, z));
	}
}

void triangulate_points(Mat M0, Mat M1, vector<Point2f> points1, vector<Point2f> points2, vector<Point3f> *points3d)
{
	triangulatePoints(M0, M1, init, points2, triOut);
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
			k++;
			circle(frame, points2[i], 3, Scalar(0, 255, 0), -1, 8);
		}
		points1.resize(k);
		points2.resize(k);
		init.resize(k);
	}

	if (points1.size() > 8) {

		float f = K.at<double>(0, 0);

		Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));

		Mat E = findEssentialMat(init, points2, f, pp, RANSAC, 0.99, 2.0, mask);
		// save E for previous and calculate current from prev N translations.

		int inliers = recoverPose(E, init, points2, R, T, f, pp);
		// construct cameramatrix and multiply for each keyframefinding the total translation.
		// maybe look at bundle adjustment ad run it in other thread between keyframe translations.

		// try to compile viz again and use it to plot 3d point cloud.

		//printf("recoverPose inliers: %d\n", inliers);
		//W = [0 1 0; -1 0 0; 0 0 1];

		//printf("x: %.2f y: %.2f z: %.2f\n", T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0));
		totalT = totalT + T;
		// remove outliers
		//Put into a function and add 3d points
		size_t k;
		for (size_t i = k = 0; i < mask.rows; i++) {
			if (!(int)mask.at<uchar>(i, 0))
				continue;
			points2[k] = points2[i];
			points1[k] = points1[i];
			init[k] = init[i];
			k++;
			//circle(frame, points2[i], 3, Scalar(0, 255, 0), -1, 8);
			if (!rdpoints){
				line(frame, init[i], points2[i], Scalar(0, 255, 0));
			}
		}
		points1.resize(k);
		points2.resize(k);
		init.resize(k);

		// comapre solution to pnp pose solution.

		//poseplot.setTo(cv::Scalar(0, 0, 0));

		char buff1[50];
		sprintf(buff1, "%+.2f %+.2f %+.2f", R.at<double>(0, 1), R.at<double>(1, 0), R.at<double>(0, 2));
		int fontFace = QT_FONT_NORMAL;
		double fontScale = 0.5f;
		int thickness = 1;
		string text(buff1);

		putText(poseplot, text, Point(0, 20), fontFace, fontScale, Scalar::all(255), thickness, 8);

		//printf("x: %.2f y: %.2f z: %.2f\n", TT.at<double>(0, 0), TT.at<double>(1, 0), TT.at<double>(2, 0));
		//circle(poseplot, Point(200 + totalT.at<double>(0, 0), 200 + totalT.at<double>(1, 0)), 2, Scalar(0, 255, 0));

		//printf("%d\n",t.size());
	}

	//kpts2.clear();
	//kpts1.clear();
	imshow("cam", frame);

	int key = waitKey(15);
	//points1.size() < initPoints / 2 ||

	if (key == ' ') {
		if (started && !rdpoints) {
			rdpoints = 1;
			// move to triangulate function
			// calculate ||P11-P12|| / ||P21 - P22|| as a scale ratio between initialization and current essential matrix
			// read about keyframes.

			Mat M1;
			hconcat(R, T, M1);

			triangulate_points(K*M0, K*M1, init, points2, &init3dpoints);

			Mat rR;
			Rodrigues(R, rR);
			Mat fR = Mat::eye(3, 3, CV_64F);

			Rodrigues(fR, rR);
			Mat fT = Mat::zeros(3, 1, CV_64F);
			projectPoints(init3dpoints, rR, fT, K, noArray(), reprojected);
			for (int i = 0; i < reprojected.size(); i++) {
				printf("(%f, %f)\n", reprojected[i].x, reprojected[i].y);
				circle(poseplot, Point2f(init[i].x, init[i].y), 3, Scalar(255, 0, 0));
				circle(poseplot, Point2f(points2[i].x, points2[i].y), 3, Scalar(0, 0, 255));
				circle(poseplot, Point2f(reprojected[i].x, reprojected[i].y), 1, Scalar(0, 255, 0));
			}
		}
	}

	std::swap(points2, points1);
	cv::swap(prevGray, grey);

	if (key == ' ') {
		started = 1;
		// features and keypoints for object
		img1 = grey.clone();
		kpts1.clear();
		init.clear();
		goodFeaturesToTrack(img1, points1, MAX_FEATURES, 0.01, 20, Mat(), 3, 0, 0.04);
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
