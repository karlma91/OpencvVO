/*#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio


void startVO(void)
{
	Mat img1 = imread("C:/opencv/sources/samples/data/graf1.png", IMREAD_GRAYSCALE);
	Mat img2 = imread("C:/opencv/sources/samples/data/graf3.png", IMREAD_GRAYSCALE);

	Mat homography;
	FileStorage fs("C:/opencv/sources/samples/data/H1to3p.xml", FileStorage::READ);
	fs.getFirstTopLevelNode() >> homography;

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);

	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}

	/*for (unsigned i = 0; i < matched1.size(); i++) {
	Mat col = Mat::ones(3, 1, CV_64F);
	col.at<double>(0) = matched1[i].pt.x;
	col.at<double>(1) = matched1[i].pt.y;

	col = homography * col;
	col /= col.at<double>(2);
	double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
	pow(col.at<double>(1) - matched2[i].pt.y, 2));

	if (dist < inlier_threshold) {
	int new_i = static_cast<int>(inliers1.size());
	inliers1.push_back(matched1[i]);
	inliers2.push_back(matched2[i]);
	good_matches.push_back(DMatch(new_i, new_i, 0));
	}
	}*/
/*
	vector<Point2f> points1; KeyPoint::convert(matched1, points1);
	vector<Point2f> points2; KeyPoint::convert(matched2, points2);
	Mat fun = findFundamentalMat(points1, points2, FM_RANSAC);
	Mat E = findHomography(points1, points2, RANSAC);
	Mat res;
	Mat iE;
	invert(E, iE);

	warpPerspective(img2, img2, iE, img2.size());

	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);

	cout << homography << endl;
	cout << E << endl;
	cout << fun << endl;
	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
	cout << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
	cout << "# Matches:                            \t" << matched1.size() << endl;
	cout << "# Inliers:                            \t" << inliers1.size() << endl;
	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
	cout << endl;

	imshow("res", res);

}

int main(void)
{

	startVO();
	waitKey(0);

	return 0;
}*/
