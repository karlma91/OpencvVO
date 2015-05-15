#ifndef _A_H // must be unique name in the project
#define _A_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

int RunOrb();
int PyrLKTest();
int PyrLKod();
int cameraPose();
int PNPOrb();
int VOOrb();
int KFVOOrb();

#endif 

