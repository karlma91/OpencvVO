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


int RunOrb() {

    const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
    const double ransac_thresh = 2.5f; // RANSAC inlier threshold
    const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
    const int stats_update_period = 10; // On-screen statistics are updated every 10 frames
    const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
    const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

    Ptr<ORB> detector;
    Ptr<ORB> descriptor;

    //Ptr<AKAZE> akaze = AKAZE::create();
    //akaze->set("threshold", akaze_thresh);
    //Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();

    Mat img1;
    Mat grey;
    Mat frame, img_matches;

    vector<KeyPoint> kpts1, kpts2;
    std::vector<DMatch> matches;
    Mat desc1, desc2;
    int picture_taken = 0;
    BFMatcher matcher(NORM_HAMMING);


    //FlannBasedMatcher matcher;

    VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

    if (!stream1.isOpened()) { //check if video device has been initialised
        cout << "cannot open camera";
    }

    img1 = imread("C:/opencv/sources/samples/data/box.png", IMREAD_GRAYSCALE);

    detector = ORB::create();
    detector->setMaxFeatures(250);
    descriptor = detector;


    // features and keypoints for object
    detector->detect(img1, kpts1);

    descriptor->compute(img1, kpts1, desc1);

    //unconditional loop
    while (true) {

        stream1.read(frame);

        resize(frame, frame, Size(frame.cols*0.8, frame.rows*0.8));

        cvtColor(frame, grey, COLOR_BGR2GRAY);
        //equalizeHist(grey, grey);


        img_matches = frame.clone();

        detector->detect(grey, kpts2);
        descriptor->compute(grey, kpts2, desc2);

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

                //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
                std::vector< DMatch > good_matches;
                std::vector<KeyPoint> matched1, matched2;
                for (int i = 0; i < desc1.rows; i++)
                {
                    if (matches[i].distance < 4 * min_dist)
                    {
                        good_matches.push_back(matches[i]);
                        matched1.push_back(kpts1[matches[i].queryIdx]);
                        matched2.push_back(kpts2[matches[i].trainIdx]);
                    }

                }

                vector<Point2f> points1; KeyPoint::convert(matched1, points1);
                vector<Point2f> points2; KeyPoint::convert(matched2, points2);

                if (points1.size() > 3) {

                    //Mat fun = findFundamentalMat(points1, points2, FM_RANSAC);
                    Mat mask;
                    Mat H = findHomography(points1, points2, RANSAC, 2.0, mask, 2000, 0.995);
                    if (H.dims > 0){
                        //-- Get the corners from the image_1 ( the object to be "detected" )
                        std::vector<Point2f> obj_corners(4);
                        obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
                        obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
                        std::vector<Point2f> scene_corners(4);
                        float det = determinant(H);

                        if (det > 0.001){

                            perspectiveTransform(obj_corners, scene_corners, H);

                            line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
                            line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
                            line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
                            line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

                        }
                    }
                }
            }
        }

        imshow("cam", img_matches);
        int key = waitKey(15);

        if (key == ' ') {
            // features and keypoints for object
            img1 = grey.clone();
            resize(img1, img1, Size(img1.cols*0.8, img1.rows*0.8));

            vector<Point2f> points;
            TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
            goodFeaturesToTrack(img1, points, 500, 0.01, 30, Mat(), 3, 0, 0.04);
            //Size subPixWinSize(10, 10);
            //cornerSubPix(img1, points, subPixWinSize, Size(-1, -1), termcrit);
            for (size_t i = 0; i < points.size(); i++) {
                kpts1.push_back(cv::KeyPoint(points[i], 1.f));
            }

            //equalizeHist(img1, img1);
            Mat temp;
            drawKeypoints(img1, kpts1, temp, Scalar(0, 0, 255));
            imshow("obj", temp);
            //resize(img1, img1, Size(img1.cols/2, img1.rows));
           // detector->detect(img1, kpts1);
            descriptor->compute(img1, kpts1, desc1);
            drawKeypoints(img1, kpts1, img1);
        }
        else if (key == 'q'){
            break;
        }

    }
    return 0;
}