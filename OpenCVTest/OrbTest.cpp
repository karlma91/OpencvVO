#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/viz.hpp>

#include "Tests.h"

using namespace cv;
using namespace std;

struct keyframe {
	Mat img; // image
	Mat desc; // descriotors for image
	vector<KeyPoint> kpts; // keypoints for image
	Mat H; // homograpy from previous keyframe
	Vec3d imgpos; // position for image
	Mat C; // camera matrix
};

static int MAX_FEATURES = 500;
int RunOrb() {

	Ptr<ORB> detector = ORB::create();
	Ptr<ORB> descriptor = ORB::create();
	//Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
	//Ptr<BRISK> descriptor = BRISK::create(10, 6, 1);
	//Ptr<BRISK> detector = BRISK::create(10, 6, 1);
	//Ptr<AKAZE> descriptor = AKAZE::create();
	//Ptr<AKAZE> detector = AKAZE::create();
	//akaze->set("threshold", akaze_thresh);

	Mat TT, RR,K;
	Mat img1;
	Mat grey;
	Mat frame, img_matches;

	viz::Viz3d myWindow("VIZ");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	vector<KeyPoint> kpts2;
	std::vector<DMatch> matches;
	Mat desc2;
	int picture_taken = 0;
	BFMatcher matcher(NORM_HAMMING);

	//FlannBasedMatcher matcher;

	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	FileStorage fs("C:/Users/Karlmka/Dropbox/unik4690/Kamerakalibrering/camera.yml", FileStorage::READ);
	fs["camera_matrix"] >> K;
	fs.release();

	vector<keyframe> keyframes;

	while (true) {

		stream1.read(frame);

		cvtColor(frame, grey, COLOR_BGR2GRAY);
		//equalizeHist(grey, grey);

		img_matches = frame.clone();

		//detector->detect(grey, kpts2);
		kpts2.clear();
		vector<Point2f> points1;
		TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 200, 0.03);
		Size subPixWinSize(10, 10);
		Size winSize(15, 15); // 31
		goodFeaturesToTrack(grey, points1, MAX_FEATURES, 0.01, 20, Mat(), 3, 0, 0.04);
		cornerSubPix(grey, points1, subPixWinSize, Size(-1, -1), termcrit);
		for (size_t i = 0; i < points1.size(); i++) {
			kpts2.push_back(cv::KeyPoint(points1[i], 1.f));
		}

		KeyPointsFilter::retainBest(kpts2, MAX_FEATURES);
		descriptor->compute(grey, kpts2, desc2);
		if (desc2.rows > 5 && keyframes.size() > 0) {
			keyframe key = keyframes[keyframes.size() - 1];

			matcher.match(key.desc, desc2, matches);

			if (matches.size() > 0) {
				double max_dist = 0; double min_dist = 100;
				double avg_img_dist = 0;
				//-- Quick calculation of max and min distances between keypoints
				for (int i = 0; i < key.desc.rows; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
				std::vector< DMatch > good_matches;
				std::vector<KeyPoint> matched1, matched2;
				for (int i = 0; i < key.desc.rows; i++)
				{
					if (matches[i].distance < 40 * min_dist)
					{
						good_matches.push_back(matches[i]);
						matched1.push_back(key.kpts[matches[i].queryIdx]);
						matched2.push_back(kpts2[matches[i].trainIdx]);
						avg_img_dist += norm(key.kpts[matches[i].queryIdx].pt - kpts2[matches[i].trainIdx].pt);
					}

				}
				avg_img_dist /= matched1.size();
				int k = 0;
				for (int i = 0; i < matched1.size(); i++){
					if (norm(matched1[i].pt - matched2[i].pt) < avg_img_dist * 2) {
						matched1[k] = matched1[i];
						matched2[k] = matched2[i];
						k++;
					}
				}
				matched1.resize(k);
				matched2.resize(k);

				vector<Point2f> points1; KeyPoint::convert(matched1, points1);
				vector<Point2f> points2; KeyPoint::convert(matched2, points2);

				if (points1.size() > 3) {

					//Mat fun = findFundamentalMat(points1, points2, FM_RANSAC);
					Mat mask;
					Mat H = findHomography(points1, points2, RANSAC, 2.0, mask, 2000, 0.995);
					if (!H.empty() && determinant(H) > 0.001){
						vector<Mat> R;
						vector<Mat> t;
						vector<Mat> N;

						decomposeHomographyMat(H, K, R, t, N);

						Mat M0 = Mat::eye(3, 4, CV_64F);

						Mat M1;
						hconcat(R[0], t[0], M1);
						TT = t[0];
						int idx = 0;
						if (t.size() > 1){
							for (int i = 0; i < 4; i++) {
								if (N[i].at<double>(2, 0) <= N[idx].at<double>(2, 0)){
									idx = i;
								}
							}
							TT = t[idx];
							RR = R[idx];
						}

						//-- Get the corners from the image_1 ( the object to be "detected" )
						std::vector<Point2f> obj_corners(4);
						obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
						obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
						std::vector<Point2f> scene_corners(4);
						perspectiveTransform(obj_corners, scene_corners, H);
						line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
						line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
						line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
						line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
					}
				}
			}
		}
		
		//drawKeypoints(img_matches, kpts2, img_matches,Scalar(255,0,0));
		imshow("cam", img_matches);
		int key = waitKey(15);

		if (key == ' ') {
			img1 = grey.clone();
			//detector->detect(img1, kpts1);
			keyframe key = keyframe();
			key.img = grey.clone();
			if (keyframes.size() == 0){
				key.imgpos = Vec3b(0, 0, 2);
				key.C = Mat::eye(3, 4, CV_64F);
			}

			vector<Point2f> points1, points2;
			TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 200, 0.03);
			Size subPixWinSize(10, 10);
			Size winSize(15, 15); // 31
			goodFeaturesToTrack(key.img, points1, MAX_FEATURES, 0.01, 20, Mat(), 3, 0, 0.04);
			cornerSubPix(key.img, points1, subPixWinSize, Size(-1, -1), termcrit);
			for (size_t i = 0; i < points1.size(); i++) {
				key.kpts.push_back(cv::KeyPoint(points1[i], 1.f));
			}
			descriptor->compute(key.img, key.kpts, key.desc);
			keyframes.push_back(key);
			drawKeypoints(img1, key.kpts, img1);
			imshow("obj", img1);
		}
		else if (key == 'q'){
			break;
		}
		if (TT.rows > 0) {
			Mat fD = (Mat_<double>(3, 1) << 0, 0, 1);
			Mat tmp = TT + (RR.inv())*fD;// R*fD;
			Vec3d cam_pos(TT), cam_focal_point(tmp), cam_y_dir(0.0f, -1.0f, 0.0f);
			/// We can get the pose of the cam using makeCameraPose
			Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

			viz::WCameraPosition cpw(0.5); // Coordinate axes
			viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
			myWindow.showWidget("CPW", cpw, cam_pose);
			myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
			for (int i = 0; i < keyframes.size(); i++) {
				myWindow.showWidget("img1" + i, viz::WImage3D(keyframes[i].img, Size2d(1.0, 1.0), Vec3d(0, i, 2.0), Vec3d(0.0, 0.0, 1.0), Vec3d(0.0, 1.0, 0.0)));
			}
		}

		myWindow.spinOnce(1, true);

	}
	return 0;
}