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
#include <opencv2/viz.hpp>
#include "Tests.h"

static int MAX_FEATURES = 300;
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
static VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

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
				int k = 0;
				for (int i = 0; i < init.size(); i++) {
					double dist = norm(init[i] - points2[i]);
					//printf("%f\n", dist);
					if (dist > avg_dist){
						continue;
					}
					points2[k] = points2[i];
					init[k] = init[i];
					k++;
				}
				points2.resize(k);
				init.resize(k);

				if (good_matches.size() > 10 && init.size() > 6) {

					float f = K.at<double>(0, 0);
					Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
					E = findEssentialMat(init, points2, f, pp, RANSAC, 0.999, 1.0, mask);
					int inliers = recoverPose(E, init, points2, R, T, f, pp, mask);
					if (inliers > 10){
						printf("%d\n", inliers);
						hconcat(R, T, M1);
						cv::Mat row = cv::Mat::zeros(1, 4, CV_64F);
						row.at<double>(0, 3) = 1;
						M1.push_back(row);
						//print(M1);
						totalT = totalT*M1;

						Mat rot;
						totalT(cv::Range(0, 3), cv::Range(0, 3)).copyTo(rot);
						Mat rotv;
						Rodrigues(rot, rotv);
						poseplot(Range(0, 100), Range(0, 300)) = 0;

						char buff1[50];
						int fontFace = QT_FONT_NORMAL;
						double fontScale = 0.5f;
						int thickness = 1;
						sprintf(buff1, "x:%+.1f y:%+.1f z:%+.1f", rotv.at<double>(0, 0) * (180 / CV_PI), 
							(rotv.at<double>(1, 0) / CV_PI) * 180, (rotv.at<double>(2, 0) / CV_PI) * 180);
						string text(buff1);
						putText(poseplot, text, Point(0, 20), fontFace, fontScale, Scalar::all(255), thickness, 8);

						circle(poseplot, Point(100 + totalT.at<double>(0, 3) * 3, 100 + totalT.at<double>(1, 3)) * 3, 2, Scalar(0, 255, 0));
					}
					kpts1.clear();
					for (int i = 0; i < kpts2.size(); i++) {
						kpts1.push_back(kpts2[i]);
					}
					desc2.copyTo(desc1);
				}
			}
		}
	}
	if (mask.rows > 0) {
		for (size_t i = 0; i < min(init.size(), points2.size()); i++) {
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

	int key = waitKey(15);
	if (key == ' '|| kpts1.size() < 30) {
		started = 1;
		kpts1.clear();
		detector->detect(grey, kpts1);
		cv::KeyPointsFilter::retainBest(kpts1, MAX_FEATURES);
		descriptor->compute(grey, kpts1, desc1);
		KeyPoint::convert(kpts1, points1);
		poseplot.setTo(cv::Scalar(0, 0, 0));
		totalT = Mat::eye(4, 4, CV_64F);
	}
	else if (key == 'q') {
		return;
	}

	grey.copyTo(prevGray);

	imshow("pose", poseplot);
}

int VOOrb() {
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

	while (true) {
		loop();
	}
	return 0;
}
