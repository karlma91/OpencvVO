
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/viz.hpp>
#include "Tests.h"
#include <unistd.h>

using namespace cv;
using namespace std;

void extractRTfromH(Mat& H, Mat& Rot, Mat& Trans);
void trackNextFrame();
void updateTotalT(Mat& totalT, Mat &R, Mat &t);
void initNewFeatures();
void plotCorners();

vector<Point2f> points1, points2, init;
TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 200, 0.03);
Size subPixWinSize(10, 10);
Size winSize(15, 15); // 31

static Mat img1;
static Mat grey;
static Mat frame, img_matches;
static Mat temp, prevGray;
static Mat K;
static Mat H;

static int initialPointCount = 0;

static Mat totalT = Mat::eye(4, 4, CV_64F);
static Mat totalH  = Mat::eye(3, 3, CV_64F);;
static Mat poseplot = Mat::zeros(400, 400, CV_8UC3);

static int MAX_FEATURES = 500;
static vector<KeyPoint> kpts1, kpts2;
static std::vector<DMatch> matches;

static VideoCapture stream1(1);   //0 is the id of video device.0 if you have only one camera.

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

  Mat desc1, desc2;
  int picture_taken = 0;
  //BFMatcher matcher(NORM_HAMMING);
  int initPoints;

  Mat R = Mat::eye(3, 3, CV_64F);
  Mat t(3, 1, CV_64F);

  //Mat totalR = Mat::eye(3, 3, CV_64F);
  //Mat totalT(3, 1, CV_64F);
  
  Mat sH = Mat::eye(3, 3, CV_64F);

  Mat tempH =  Mat::eye(3, 3, CV_64F);

  static viz::Viz3d myWindow("VIZ");

  if (!stream1.isOpened()) { //check if video device has been initialised
    cout << "cannot open camera" << endl;
  }

  FileStorage fs("../camera.yml", FileStorage::READ);
  fs["camera_matrix"] >> K;
  fs.release();

  print(K);

  myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
  //unconditional loop
  while (true) {
    
    stream1.read(frame);
    
    //resize(frame, frame, Size(frame.cols*0.8, frame.rows*0.8));
    cvtColor(frame, grey, COLOR_BGR2GRAY);		
    
    if (!points1.empty()){      
      trackNextFrame();
    }

    // Calculate homography
    if (points1.size() > 8) {
      Mat mask;
      H = findHomography(init, points2, RANSAC, 3.0, mask,2000,0.98);

      if (!H.empty() && determinant(H) > 0.001){
        extractRTfromH(H, R, t);
        plotCorners();

      }

    }
    //kpts2.clear();
    //kpts1.clear();
    imshow("cam", frame);

    std::swap(points2, points1);
    cv::swap(prevGray, grey);

    int key = waitKey(15);

    if ( key == ' ') {
      // features and keypoints for object
      sH = Mat::eye(3, 3, CV_64F);

      initNewFeatures();
      
      //Mat temp;
      //drawKeypoints(img1, kpts1, temp, Scalar(0, 0, 255));
      //imshow("obj", temp);
      //drawKeypoints(img1, kpts1, img1);
    }
    else if (key == 'q') {
      break;
    }

    /**
       Store current camera position/rotation and reset feature points
       to track
    */
   
    if (!t.empty() && img1.rows > 0) {
      //      Mat R = totalT.rowRange(0,2).colRange(0,);

      print(t);
      cout << endl;

      Mat totT;
      totalT.copyTo(totT);
      updateTotalT(totT, R, t);

      Mat R = totT.rowRange(0,3).colRange(0,3);
      Mat t = totT.col(3).rowRange(0,3);

      cout << "t vector is: "<< endl;
      print(t);
      cout << endl;

      cout << "R mat is: "<< endl;
      print(R);
      cout << endl;
      
      Mat fD = (Mat_<double>(3, 1) << 0, 0, 1);
      Mat tmp = t + (R.inv())*fD;// R*fD;
      Vec3d cam_pos(t);
      Vec3d cam_focal_point(tmp);
      Vec3d cam_y_dir(0.0f, -1.0f, 0.0f);
      /// We can get the pose of the cam using makeCameraPose
      Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

      viz::WCameraPosition cpw(0.5); // Coordinate axes
      viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
      myWindow.showWidget("CPW", cpw, cam_pose);
      myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
      myWindow.showWidget("img1", viz::WImage3D(img1, Size2d(1.0, 1.0), Vec3d(0, 0.0, 2.0), Vec3d(0.0, 0.0, 1.0), Vec3d(0.0, 1.0, 0.0)));
    }

    // Couldn't find homography, so reset the feature points. 
    /* if (H.empty()) {
      initNewFeatures();
    }
    else */if(points1.size() < initialPointCount / 2) {
      tempH = H*totalH;

      
      updateTotalT(totalT, R, t);
      totalH = tempH; // Update total homography transform. 
        
      initNewFeatures();
      //std::swap(points2, points1);
    }
    
    myWindow.spinOnce(1, true);
    imshow("pose", poseplot);
  }
  return 0;
}

void initNewFeatures() {
  img1 = grey.clone();
  cout << "INITIALIZING NEW POINTS TO TRACK!" << endl;
  kpts1.clear();
  init.clear();
  points1.clear();
  points2.clear();
  goodFeaturesToTrack(img1, points1, MAX_FEATURES, 0.01, 20, Mat(), 3, 0, 0.04);
  cornerSubPix(img1, points1, subPixWinSize, Size(-1, -1), termcrit);
  
  
  for (size_t i = 0; i < points1.size(); i++) {
    kpts1.push_back(cv::KeyPoint(points1[i], 1.f));
    init.push_back(Point2f(points1[i]));
  }

  initialPointCount = init.size();
}

void updateTotalT(Mat& totalT, Mat &R, Mat &t) {
  Mat M1 = Mat::eye(3, 4, CV_64F);
  cv::Mat row = cv::Mat::zeros(1, 4, CV_64F);
  hconcat(R, t, M1);  
  row.at<double>(0, 3) = 1;
  M1.push_back(row);
  //print(M1);
  totalT = totalT*M1;
}

void trackNextFrame() {
  vector<uchar> status;
  vector<float> err;
    
  calcOpticalFlowPyrLK(prevGray, grey, points1, points2, status, err, winSize,
                       3, termcrit, 0, 0.001);

  size_t i, k;

  // Resize feature point vectors. 
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

void getH() {

}

void extractRTfromH(Mat& H, Mat& Rot, Mat& Trans) {
  vector<Mat> R;
  vector<Mat> t;
  vector<Mat> N;
  
  decomposeHomographyMat(H, K, R, t, N);

  Mat M0 = Mat::eye(3, 4, CV_64F);
  Mat M1;
  hconcat(R[0], t[0], M1);

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

        
  poseplot.setTo(cv::Scalar(0, 0, 0));
  //Mat T = pose.col(3);
  Trans = t[0];
  int idx = 0;
  if (t.size() > 1){
    for (int i = 0; i < 4; i++) {
      if (N[i].at<double>(2, 0) <= N[idx].at<double>(2, 0)){
        idx = i;
      }
    }
    Trans = t[idx];
    Rot = R[idx];
  }

  char buff1[50];
  int fontFace = QT_FONT_NORMAL;
  double fontScale = 0.5f;
  int thickness = 1;
  for (int i = 0; i < t.size(); i++) {
    sprintf(buff1, "%d:[%+.1f %+.1f %+.1f]  [%+.1f %+.1f %+.1f] %+.2f\n", i,
            t[i].at<double>(0, 0), t[i].at<double>(1, 0), t[i].at<double>(2, 0),
            N[i].at<double>(0, 0), N[i].at<double>(1, 0), N[i].at<double>(2, 0), c1);
    string text(buff1);

    putText(poseplot, text, Point(0, 20+i*20), fontFace, fontScale, Scalar::all(255), thickness, 8);
  }

  //printf("x: %.2f y: %.2f z: %.2f\n", Trans.at<double>(0, 0), Trans.at<double>(1, 0), Trans.at<double>(2, 0));
  circle(poseplot, Point(200 + Trans.at<double>(0, 0) * 100, 200 + Trans.at<double>(1, 0) * 100), 2, Scalar(0, abs(Trans.at<double>(2, 0))*150 + 100, 0));

  //printf("%d\n",t.size());
}

void plotCorners() {
  
  
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
  obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform(obj_corners, scene_corners, H*totalH);

  line(frame, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
  line(frame, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
  line(frame, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
  line(frame, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
}
