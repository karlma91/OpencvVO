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


void cameraPoseFromHomography(const Mat& H, Mat& pose)
{
	pose = Mat::eye(3, 4, CV_32FC1);      // 3x4 matrix, the camera pose
	float norm1 = (float)norm(H.col(0));
	float norm2 = (float)norm(H.col(1));
	float tnorm = (norm1 + norm2) / 2.0f; // Normalization value

	Mat p1 = H.col(0);       // Pointer to first column of H
	Mat p2 = pose.col(0);    // Pointer to first column of pose (empty)

	cv::normalize(p1, p2);   // Normalize the rotation, and copies the column to pose

	p1 = H.col(1);           // Pointer to second column of H
	p2 = pose.col(1);        // Pointer to second column of pose (empty)

	cv::normalize(p1, p2);   // Normalize the rotation and copies the column to pose

	p1 = pose.col(0);
	p2 = pose.col(1);

	Mat p3 = p1.cross(p2);   // Computes the cross-product of p1 and p2
	Mat c2 = pose.col(2);    // Pointer to third column of pose
	p3.copyTo(c2);       // Third column is the crossproduct of columns one and two

	pose.col(3) = H.col(2) / tnorm;  //vector t [R|t] is the last column of pose
}

int PyrLKTest() {


	Mat img1;
	Mat grey;
	Mat frame, img_matches;
	Mat temp, prevGray;

	int MAX_FEATURES = 500;
	vector<KeyPoint> kpts1, kpts2;
	std::vector<DMatch> matches;
	Mat desc1, desc2;
	int picture_taken = 0;
	//BFMatcher matcher(NORM_HAMMING);
	int initPoints;

	Mat sH = Mat::eye(3, 3, CV_64F);

	Mat K;


	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	FileStorage fs("C:/Users/Karlmka/Dropbox/unik4690/Kamerakalibrering/camera.yml", FileStorage::READ);
	fs["camera_matrix"] >> K;
	fs.release();

	print(K);
	vector<Point2f> points1, points2, init;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 200, 0.03);
	Size subPixWinSize(10, 10);
	Size winSize(15, 15); // 31

	Mat poseplot = Mat::zeros(400, 400, CV_8UC3);
	//unconditional loop
	while (true) {

		stream1.read(frame);
		//resize(frame, frame, Size(frame.cols*0.8, frame.rows*0.8));
		cvtColor(frame, grey, COLOR_BGR2GRAY);		

		vector<uchar> status;
		vector<float> err;

		if (!points1.empty()){
			calcOpticalFlowPyrLK(prevGray, grey, points1, points2, status, err, winSize,
				3, termcrit, 0, 0.001);

			size_t i, k;
			for (i = k = 0; i < points2.size(); i++)
			{

				if (!status[i])
					continue;

				points2[k] = points2[i];
				points1[k] = points1[i];
				init[k] = init[i];
				k++;
				//kpts1.push_back(KeyPoint(points1[i], 1.f));
				//kpts2.push_back(KeyPoint(points2[i], 1.f));
				
				circle(frame, points2[i], 3, Scalar(0, 255, 0), -1, 8);

				//circle(frame, points1[i], 3, Scalar(255, 0, 0), -1, 8);
			}
			points1.resize(k);
			points2.resize(k);
			init.resize(k);

		}

		if (points1.size() > 8) {
			Mat mask;
			Mat H = findHomography(init, points2, RANSAC, 3.0, mask,2000,0.98);

			//warpPerspective(frame, frame, H, Size(640, 480));

			if (!H.empty() && determinant(H) > 0.001){
				vector<Mat> R;
				vector<Mat> t;
				vector<Mat> N;

				decomposeHomographyMat(H, K, R, t, N);


				Mat M0 = Mat::eye(3, 4, CV_64F);

				Mat M1;
				hconcat(R[0], t[0], M1);

				/*size_t i, k;
				for (i = k = 0; i < mask.rows; i++)
				{

					if (!(int)mask.at<uchar>(i,0))
						continue;

					points2[k] = points2[i];
					points1[k] = points1[i];
					init[k] = init[i];
					k++;
					//kpts1.push_back(KeyPoint(points1[i], 1.f));
					//kpts2.push_back(KeyPoint(points2[i], 1.f));

					circle(frame, points2[i], 3, Scalar(0, 255, 0), -1, 8);

					//circle(frame, points1[i], 3, Scalar(255, 0, 0), -1, 8);
				}
				points1.resize(k);
				points2.resize(k);
				init.resize(k);*/

				Mat pointa(2, 1, CV_64F);
				pointa.at<double>(0, 0) = init[0].x;
				pointa.at<double>(1, 0) = init[0].y;

				Mat pointb(2, 1, CV_64F);
				pointb.at<double>(0, 0) = points2[0].x;
				pointb.at<double>(1, 0) = points2[0].y;

				Mat triOut;
				triangulatePoints(M0, M1, pointa, pointb, triOut);

				//print(triOut);

				double c1 = triOut.at<double>(2, 0) * triOut.at<double>(3, 0);


				//printf("c1: %.2f\n", c1);
				//printf("Start\n");
				//for (int i = 0; i < t.size(); i++) {
				//	print(t[i]);
				//}
				//printf("END\n");

				//Mat F = findFundamentalMat(init, points2);
				//Mat kt;
				//transpose(K, kt);
				//Mat E = kt * F * K;
				//Mat w, u, v;
				//SVD::compute(E, w, u, v);
				//print(K);


				//float f = K.at<double>(0, 0);
				//float f = 1;

				//Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
				//Point2f pp(0, 0);
				//Mat mask;
				//Mat E = findEssentialMat(init, points2, f, pp, RANSAC, 0.999, 1.0, mask);
				//print(E);
				//Mat R;
				//Mat T;
				//recoverPose(E, init, points2, R, T, f, pp, mask);
				//W = [0 1 0; -1 0 0; 0 0 1];
				//printf("x: %.2f y: %.2f z: %.2f\n", T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0));

				poseplot.setTo(cv::Scalar(0, 0, 0));
				//Mat T = pose.col(3);
				Mat TT = t[0];
				int idx = 0;
				if (t.size() > 1){
					for (int i = 0; i < 4; i++) {
						if (N[i].at<double>(2, 0) >= N[idx].at<double>(2, 0)){
							idx = i;
						}
					}
					TT = t[idx];
				}

				char buff1[50];
				int fontFace = QT_FONT_NORMAL;
				double fontScale = 0.5f;
				int thickness = 1;
				for (int i = 0; i < 4; i++) {
					sprintf(buff1, "%d:[%+.1f %+.1f %+.1f]  [%+.1f %+.1f %+.1f] %+.2f\n", i,
						t[i].at<double>(0, 0), t[i].at<double>(1, 0), t[i].at<double>(2, 0),
						N[i].at<double>(0, 0), N[i].at<double>(1, 0), N[i].at<double>(2, 0), c1);
					string text(buff1);

					putText(poseplot, text, Point(0, 20+i*20), fontFace, fontScale, Scalar::all(255), thickness, 8);
				}

				//printf("x: %.2f y: %.2f z: %.2f\n", TT.at<double>(0, 0), TT.at<double>(1, 0), TT.at<double>(2, 0));
				circle(poseplot, Point(200 + TT.at<double>(0, 0) * 100, 200 + TT.at<double>(1, 0) * 100), 2, Scalar(0, abs(TT.at<double>(2, 0))*150 + 100, 0));

				//printf("%d\n",t.size());

				std::vector<Point2f> obj_corners(4);
				obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
				obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
				std::vector<Point2f> scene_corners(4);

				CvPoint T = CvPoint(0, 0);


				//transpose(H,H);

				//sH = H*sH;
				perspectiveTransform(obj_corners, scene_corners, H);

				line(frame, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
				line(frame, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
				line(frame, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
				line(frame, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
			}

		}

		//kpts2.clear();
		//kpts1.clear();
		imshow("cam", frame);

		std::swap(points2, points1);
		cv::swap(prevGray, grey);

		int key = waitKey(15);
		//points1.size() < initPoints / 2 ||
		if ( key == ' ') {
			// features and keypoints for object
			sH = Mat::eye(3, 3, CV_64F);
			img1 = grey.clone();
			kpts1.clear();
			init.clear();
			goodFeaturesToTrack(img1, points1, MAX_FEATURES, 0.01, 20, Mat(), 3, 0, 0.04);
			cornerSubPix(img1, points1, subPixWinSize, Size(-1, -1), termcrit);
			for (size_t i = 0; i < points1.size(); i++) {
				kpts1.push_back(cv::KeyPoint(points1[i], 1.f));
				init.push_back(Point2f(points1[i]));
			}
			Mat temp;
			//drawKeypoints(img1, kpts1, temp, Scalar(0, 0, 255));
			//imshow("obj", temp);
			//drawKeypoints(img1, kpts1, img1);
		}
		else if (key == 'q') {
			break;
		}

		imshow("pose", poseplot);
	}
	return 0;
}