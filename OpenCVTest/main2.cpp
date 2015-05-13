/*#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	Mat img1 = imread("C:/opencv/sources/samples/data/box.png", IMREAD_GRAYSCALE);

	const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
	const double ransac_thresh = 2.5f; // RANSAC inlier threshold
	const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
	const int stats_update_period = 10; // On-screen statistics are updated every 10 frames
	const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
	const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

	Ptr<AKAZE> akaze = AKAZE::create();
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();

	//akaze->set("threshold", akaze_thresh);


	//FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	Ptr<ORB> orb = ORB::create();
	orb->setMaxFeatures(800);
	BFMatcher matcher(NORM_HAMMING);

	// features for object

	orb->detectAndCompute(img1, noArray(), kpts1, desc1);

	//unconditional loop
	while (true) {
		Mat frame, edges;
		stream1.read(frame);
		cvtColor(frame, edges, COLOR_BGR2GRAY);

		orb->detectAndCompute(edges, noArray(), kpts2, desc2);

		if (desc2.rows > 5) {
			matcher.match(desc1, desc2, matches);

			if (matches.size() > 0) {
				double max_dist = 0; double min_dist = 100;

				//-- Quick calculation of max and min distances between keypoints
				for (int i = 0; i < desc1.rows; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				//printf("-- Max dist : %f \n", max_dist);
				//printf("-- Min dist : %f \n", min_dist);

				//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
				std::vector< DMatch > good_matches;
				std::vector<KeyPoint> matched1, matched2;
				for (int i = 0; i < desc1.rows; i++)
				{
					if (matches[i].distance < 3 * min_dist)
					{
						good_matches.push_back(matches[i]);
						matched1.push_back(kpts1[matches[i].queryIdx]);
						matched2.push_back(kpts2[matches[i].trainIdx]);
					}

				}

				vector<Point2f> points1; KeyPoint::convert(matched1, points1);
				vector<Point2f> points2; KeyPoint::convert(matched2, points2);

				if (points1.size() > 3){
					//Mat fun = findFundamentalMat(points1, points2, FM_RANSAC);
					Mat H = findHomography(points1, points2, RANSAC);
					if (H.dims > 0){
						//-- Get the corners from the image_1 ( the object to be "detected" )
						std::vector<Point2f> obj_corners(4);
						obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
						obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
						std::vector<Point2f> scene_corners(4);
						float det = determinant(H);

						if (det > 0.001){

							perspectiveTransform(obj_corners, scene_corners, H);

							Mat img_matches;
							drawMatches(img1, kpts1, edges, kpts2,
								good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
								vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
							//-- Draw lines between the corners (the mapped object in the scene - image_2 )
							line(img_matches, scene_corners[0] + Point2f(img1.cols, 0), scene_corners[1] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
							line(img_matches, scene_corners[1] + Point2f(img1.cols, 0), scene_corners[2] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
							line(img_matches, scene_corners[2] + Point2f(img1.cols, 0), scene_corners[3] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
							line(img_matches, scene_corners[3] + Point2f(img1.cols, 0), scene_corners[0] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);

							imshow("cam", img_matches);
						}
					}
				}
			}

			//drawKeypoints(edges, kpts1, edges, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
			//GaussianBlur(edges, edges, Size(9,9), 1.5, 1.5);
			//Canny(edges, edges, 55, 40);
		}

		if (waitKey(15) >= 0)
			break;
	}
	return 0;
}*/