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

static int MAX_FEATURES = 2000;
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
static vector<Point3f> camposes, totalmap;
static vector<Mat> keyframes;
static vector<viz::Color> colors;


static TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 200, 0.03);
static Size subPixWinSize(10, 10);
static Size winSize(15, 15); // 31

static int frames = 0;
static Mat grey;
static Mat frame, img_matches;
static Mat temp, prevGray;
static Mat triOut;
static Mat ctriout;
static Mat sH = Mat::eye(3, 3, CV_64F);
static Mat M0 = Mat::eye(3, 4, CV_64F);
static Mat M1;

static Mat poseplot = Mat::zeros(480, 640, CV_8UC3);
static Mat totalT = Mat::eye(4, 4, CV_64F);
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
static Mat pointcloud(0,0,CV_64FC3);
static Mat allColors(0, 0, CV_8UC3);
static VideoCapture stream1("C:\\Users\\Karlmka\\Dropbox\\unik4690\\20150515_172609.mp4");   //0 is the id of video device.0 if you have only one camera.

static vector<uchar> status;
static vector<float> err;
static Mat E;
//static Ptr<ORB> detector;
static Ptr<ORB> descriptor;
static Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
//Ptr<AKAZE> akaze = AKAZE::create();
//akaze->set("threshold", akaze_thresh);
//Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();

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
	triangulatePoints(CM0, CM1, poi1, poi2, triOut);
	printf("row: %d, col: %d\n", triOut.rows, triOut.cols);
	convertFromHom(triOut, points3d);
}

static void loop() {
	stream1.read(frame);
	stream1.read(frame);
	//resize(frame, frame, Size(frame.cols*0.8, frame.rows*0.8));
	cvtColor(frame, grey, COLOR_BGR2GRAY);

	if (started) {
		cv::Mat rvec(3, 1, cv::DataType<double>::type);
		cv::Mat tvec(3, 1, cv::DataType<double>::type);
		float scale = 0;

		detector->detect(grey, kpts2);
		cv::KeyPointsFilter::retainBest(kpts2, MAX_FEATURES);
		descriptor->compute(grey, kpts2, desc2);
		frames++;
		if (desc2.cols > 5 && frames > 0) {
			frames = 0;
			matcher.match(desc1, desc2, matches);
			if (matches.size() > 5) {
				double max_dist = 0; double min_dist = 1000;
				std::vector< DMatch > good_matches;
				std::vector<KeyPoint> matched1, matched2;
				for (int i = 0; i < matches.size(); i++) {
					if (matches[i].distance < 20) {
						good_matches.push_back(matches[i]);
						matched1.push_back(kpts1[matches[i].queryIdx]);
						matched2.push_back(kpts2[matches[i].trainIdx]);
					}
				}
				KeyPoint::convert(matched1, init);
				KeyPoint::convert(matched2, points2);
				float avg_dist = 0;
				for (size_t i = 0; i < good_matches.size(); i++) {
					double dist = norm(init[i] - points2[i]);
					avg_dist += dist;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}
				avg_dist = avg_dist / good_matches.size();
				/*int k = 0;
				for (int i = 0; i < init.size(); i++) {
					double dist = norm(init[i] - points2[i]);
					//printf("%f\n", dist);
					if (dist > avg_dist*2){
						continue;
					}
					points2[k] = points2[i];
					init[k] = init[i];
					k++;
				}
				points2.resize(k);
				init.resize(k);*/

				if (good_matches.size() > 10 && init.size() > 6) {

					float f = K.at<double>(0, 0);
					Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
					E = findEssentialMat(init, points2, f, pp, RANSAC, 0.995, 2.0, mask);
					int inliers = recoverPose(E, init, points2, R, T, f, pp, mask);
					if (inliers > 5){
						hconcat(R, T, M1);


						cv::Mat row = cv::Mat::zeros(1, 4, CV_64F);
						row.at<double>(0, 3) = 1;
						M1.push_back(row);
						//print(M1);
						totalT = totalT*M1;
						Point3f TTT(totalT.at<double>(0, 3), totalT.at<double>(1, 3), totalT.at<double>(2, 3));

						vector<Point2f> ny1, ny2;
						for (int i = 0; i < mask.rows; i++)
						{
							if ((int)mask.at<uchar>(i, 0) && (norm(init[i] - points2[i]) > 3)){
								ny1.push_back(init[i]);
								ny2.push_back(points2[i]);
								
							}
						}
						if (ny1.size() > 0){
							hconcat(R, T, M1);
							triangulate_points(K*M0, K*M1, ny1, ny2, &init3dpoints);
							for (int i = 0; i < (int)init3dpoints.size(); i++) {
								if (norm(init3dpoints[i]) < 50) {
									Mat gt(init3dpoints[i]);
									gt.convertTo(gt, CV_64F);
									Mat r2 = cv::Mat::zeros(1, 1, CV_64F);
									gt.push_back(r2);
									R.convertTo(R, CV_64F);
									Mat rot1 = -totalT*gt;
									Point3f gr(rot1.at<double>(0, 0), rot1.at<double>(1, 0), rot1.at<double>(2, 0));
									totalmap.push_back(TTT + gr);
									transpose(gt, gt);
									pointcloud.push_back(gt.colRange(0,3));
									Vec3b col = frame.at<Vec3b>(points2[i].y, points2[i].x);
									allColors.push_back(col);
								}
							}
						}

						camposes.push_back(Point3f(totalT.at<double>(0, 3), totalT.at<double>(1, 3), totalT.at<double>(2, 3)));
						//camposes.push_back(Point3f(totalT.at<double>(1, 3), totalT.at<double>(2, 3), 0));

						Mat rot;
						totalT(cv::Range(0, 3), cv::Range(0, 3)).copyTo(rot);
						Mat rotv;
						Rodrigues(rot, rotv);

						kpts1.clear();
						for (int i = 0; i < kpts2.size(); i++) {
							kpts1.push_back(kpts2[i]);
						}
						desc2.copyTo(desc1);
						grey.copyTo(prevGray);
					}
				}
			}
		}
	}
	if (mask.rows > 0) {
		for (size_t i = 0; i < min((int)min(init.size(), points2.size()), mask.rows); i++) {
			circle(frame, init[i], 2, Scalar(0, 255, 0));
			if ((int)mask.at<uchar>(i, 0)) {
				line(frame, init[i], points2[i], Scalar(0, 255, 0));
			}
			else{
				line(frame, init[i], points2[i], Scalar(0, 0, 255));
			}
		}
	}
	imshow("cam", frame);

	int key = waitKey(1);
	if (key == ' ' || kpts1.size() < 40) {
		started = 1;
		kpts1.clear();
		detector->detect(grey, kpts1);
		cv::KeyPointsFilter::retainBest(kpts1, MAX_FEATURES);
		descriptor->compute(grey, kpts1, desc1);
		KeyPoint::convert(kpts1, points1);
		totalT = Mat::eye(4, 4, CV_64F);
		grey.copyTo(prevGray);
	}
	else if (key == 'q') {
		return;
	}
	if (camposes.size() > 0) {
		viz::WCloud cw(camposes);
		Mat testt(totalmap);
		viz::WCloud cws(testt, allColors);
		cw.setRenderingProperty(viz::POINT_SIZE, 5);
		cws.setRenderingProperty(viz::POINT_SIZE, 3);
		myWindow.showWidget("CloudWidget1", cw);
		myWindow.showWidget("CloudWidget2", cws);
	}

	myWindow.spinOnce(1, true);
}

int KFVOOrb() {
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
