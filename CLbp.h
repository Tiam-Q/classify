#pragma once


#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>

using namespace cv;
using namespace std;


void elbp(Mat& src, Mat& dst, int radius, int neighbors);
void getUniformPatternLBPFeature(Mat src, Mat dst, int radius, int neighbors);
int getHopTimes(int n);
Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, bool normed);
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed);




