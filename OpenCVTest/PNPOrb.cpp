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

static int MAX_FEATURES = 10000;
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

  if (kpts1.size() > 6) {
    cv::Mat rvec(3, 1, cv::DataType<double>::type);
    cv::Mat tvec(3, 1, cv::DataType<double>::type);
    float scale = 0;

    detector->detect(grey, kpts2);
    cv::KeyPointsFilter::retainBest(kpts2, MAX_FEATURES);
    descriptor->compute(grey, kpts2, desc2);

    if (started && desc2.cols > 5) {
      matcher.match(desc1, desc2, matches);

      if (matches.size() > 5) {
        double max_dist = 0; double min_dist = 1000;
        //-- Quick calculation of max and min distances between keypoints
        /*for (int i = 0; i < matches.size(); i++) {
          double dist = matches[i].distance;
          if (dist < min_dist) min_dist = dist;
          if (dist > max_dist) max_dist = dist;
          }*/
        //printf("%f\n", min_dist);
        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector< DMatch > good_matches;
        std::vector<KeyPoint> matched1, matched2;
        c3dpoints.clear();
        for (int i = 0; i < matches.size(); i++) {
          if (matches[i].distance < 30) {
            good_matches.push_back(matches[i]);
            matched1.push_back(kpts1[matches[i].queryIdx]);
            matched2.push_back(kpts2[matches[i].trainIdx]);
            if (init3dpoints.size() > 0) {
              c3dpoints.push_back(init3dpoints[matches[i].queryIdx]);
            }
          }
        }
        if (good_matches.size() > 20) {
          KeyPoint::convert(matched1, init);
          KeyPoint::convert(matched2, points2);

          float f = K.at<double>(0, 0);
          Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
          E = findEssentialMat(init, points2, f, pp, RANSAC, 0.99, 3.0, mask);
          int inliers = recoverPose(E, init, points2, R, T, f, pp);
          hconcat(R, T, M1);

          min_dist = 100000;
          float avg_dist = 0;
          size_t k;
          for (size_t i = k = 0; i < mask.rows; i++) {
            if (!(int)mask.at<uchar>(i, 0))
              continue;
            points2[k] = points2[i];

            if (init3dpoints.size() > 0){
              c3dpoints[k] = c3dpoints[i];
            }
            init[k] = init[i];
            double dist = norm(init[k] - points2[k]);
            avg_dist += dist;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
            k++;

          }
          avg_dist = avg_dist / k;
          points2.resize(k);
          init.resize(k);
          if (init3dpoints.size() > 0){
            c3dpoints.resize(k);
          }
          //printf("min: %.1f, max: %.1f\n", min_dist, max_dist);
          k = 0;

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
        }

        if (c3dpoints.size() > 0) {

          hconcat(R, T, cM);
          triangulate_points(K*M0, K*cM, init, points2, &c3dpoints);
          frames = 0;
          for (int i = 0; i < min(init3dpoints.size() - 1, c3dpoints.size()-1); i++) {
            float nor1 = norm(c3dpoints[i] - c3dpoints[i + 1]);
            float nor2 = norm(init3dpoints[i] - init3dpoints[i + 1]);
            if (nor1 > 0.1 && nor2 > 0.1) {
              frames++;
              scale = scale + (nor2 / nor1);
            }
          }
          scale = scale / frames;
          circle(poseplot, Point(200 + T.at<double>(0, 0) * 4 * scale, 200 + T.at<double>(1, 0) * 4 * scale), 2, Scalar(0, 255, 0));

          /*points2.resize(c3dpoints.size());
            solvePnPRansac(c3dpoints, points2, K, noArray(), rvec, tvec, false, 200, 4);
            printf("T: \n");
            print(tvec);
            circle(poseplot, Point(200 + tvec.at<double>(0, 0) * 100, 200 + tvec.at<double>(1, 0) * 100), 2, Scalar(0, 255, 0));*/
        }
      }
    }
  }

  /*for (size_t i = 0; i < points1.size(); i++) {
    circle(frame, points1[i], 2, Scalar(0, 255, 0), -1, 8);
    }*/
  for (size_t i = 0; i < min(init.size(),points2.size()); i++) {
    line(frame, init[i], points2[i], Scalar(0, 0, 255));
  }
  imshow("cam", frame);

  int key = waitKey(15);

  if (key == 'a') {
    if (started) {
      rdpoints = 1;
      float f = K.at<double>(0, 0);
      Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));
      int inliers = recoverPose(E, init, points2, R, T, f, pp);
      hconcat(R, T, M1);
      kpts1.clear();
      for (int i = 0; i < init.size(); i++) {
        kpts1.push_back(KeyPoint(init[i], 1.f));
      }
      descriptor->compute(img1, kpts1, desc1);
      triangulate_points(K*M0, K*M1, init, points2, &init3dpoints);
    }
  }

  if (key == ' ') {
    started = 1;

    kpts1.clear();
    // features and keypoints for object
    //goodFeaturesToTrack(grey, points1, MAX_FEATURES, 0.01, 10, Mat(), 3, 0, 0.04);
    /*for (size_t i = 0; i < points1.size(); i++) {
      kpts1.push_back(cv::KeyPoint(points1[i], 1.f));
      }*/
    grey.copyTo(img1);
    detector->detect(grey, kpts1);
    cv::KeyPointsFilter::retainBest(kpts1, MAX_FEATURES);
    descriptor->compute(grey, kpts1, desc1);

    KeyPoint::convert(kpts1, points1);


  }
  else if (key == 'q') {
    return;
  }
  imshow("pose", poseplot);
}

int PNPOrb() {
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
