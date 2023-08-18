///*
//                   _ooOoo_
//                  o8888888o
//                  88" . "88
//                  (| -_- |)
//                  O\  =  /O
//               ____/`---'\____
//             .'  \\|     |//  `.
//            /  \\|||  :  |||//  \
//           /  _||||| -:- |||||-  \
//           |   | \\\  -  /// |   |
//           | \_|  ''\---/''  |   |
//           \  .-\__  `-`  ___/-. /
//         ___`. .'  /--.--\  `. . __
//      ."" '<  `.___\_<|>_/___.'  >'"".
//     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//     \  \ `-.   \_ __\ /__ _/   .-` /  /
//======`-.____`-.___\_____/___.-`____.-'======
//                   `=---='
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//         佛祖保佑       永无BUG
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//作者：2131277807@qq.com
//时间：2023\08\18
//编译器：visual studio
//环境：opencv3.4.4
//描述：想要做个opencv识别车牌的实战项目，在车牌定位过程中无法百分百准确的筛选出有车牌的区域，
//      于是对图片提取LBP特征向量然后再训练opencv svm分类模型，得到一个分类器。
//      详情参考：https://blog.csdn.net/ltj5201314/article/details/132367815?spm=1001.2014.3001.5501
//      模型的正确率在百分之九十九以上。
//*/


#include <opencv2/opencv.hpp>  
#include <opencv/cv.h>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include<iostream>  
#include<stdio.h>
#include <string>
#include <fstream>
#include "CLbp.h"


using namespace std;
using namespace cv;
using namespace cv::ml;



#define     imgRows      36      //图像行数
#define     imgCols      136     //图像列数
#define     lbpNum       256      //lbp值的种类数



//随机打乱训练集和标签
void RandomArray(Mat Train, Mat Label, int num);
//读取图片
void readImage(const string path, Mat inputImg, Mat label);
//获取图片数量
int getNum(const string path);




int main()
{
    int trainNum = 0;
    //训练集路径
    string trainPath = "C:/Users/Tiam/Desktop/classify/image/classify_train";
    //训练集总数
    trainNum = getNum(trainPath);
    //训练集
    Mat trainMat = Mat::zeros(trainNum, lbpNum*4*17, CV_32FC1);
    //训练集标签
    Mat trainLabel = Mat::zeros(trainNum, 1, CV_32SC1);
    //读取图片
    readImage(trainPath, trainMat, trainLabel);
    //打乱顺序
    RandomArray(trainMat, trainLabel, trainNum);
    //配置SVM训练器参数
    Ptr<SVM> svm = SVM::create();
    //设置SVM的模型类型：SVC是分类模型，SVR是回归模型
    svm->setType(SVM::C_SVC);
    /*设置核函数，数据由低维空间转换到高维空间后运算量会呈几何级增加，
      支持向量机能够通过核函数有效地降低计算复杂度*/
    svm->setKernel(SVM::LINEAR);
    //阶数针对POLY核函数
    svm->setDegree(0);
    //针对RBF和POLY核函数
    svm->setGamma(0);
    //偏移量针对POLY核函数
    svm->setCoef0(0);
    //惩戒因子
    svm->setC(0.1);
    //模型型别参数
    svm->setNu(0);
    svm->setP(0);
    //终止条件最小误差
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, 0.05)); 
    cout << "开始训练" << endl;
    //训练
    svm->train(trainMat, ROW_SAMPLE, trainLabel);
    //保存模型
    svm->save("svm.xml");

    cout << "训练完成" << endl;

    int testNum;
    //测试集路径
    string testPath = "C:/Users/Tiam/Desktop/classify/image/classify_test";
    //测试集数量
    testNum = getNum(testPath);
    //测试集
    Mat testMat = Mat::zeros(testNum, lbpNum * 4 * 17, CV_32FC1);
    //测试集标签
    Mat testLabel = Mat::zeros(testNum, 1, CV_32SC1);
    //读取
    readImage(testPath, testMat, testLabel);

    int count = 0;
    string modelPath = "C:/Users/Tiam/Desktop/classify/svm.xml";
    Ptr<ml::SVM>svmpre = ml::SVM::load(modelPath);
    cout << "开始预测" << endl;
    for (int i = 0; i < testNum; i++) 
    {
        Mat img;
        testMat.row(i).copyTo(img);


        int result;
        result = svmpre->predict(img);

        if (result == testLabel.at<int>(i, 0))
        {
            count++;
        }
            
    }
    cout << "预测完成" << endl;
    cout << "正确率：" << (float)count / (float)testNum << endl;

    cv::destroyAllWindows();
    getchar();
    return 0;
}




void readImage(const string path, Mat inputImg, Mat label)
{
    Mat lbpImg;
    int n = 0;
    for (int i = 0; i < 2; i++)
    {
        char folder[100];

        if (i == 0)
        {
            //负样本
            sprintf_s(folder, "%s/%s", path.c_str(), "no");
        }
        else
        {
            //正样本
            sprintf_s(folder, "%s/%s", path.c_str(), "has");
        }
        vector<cv::String> imagePathList;
        //读取路径下所有图片
        glob(folder, imagePathList);


        for (int j = 0; j < imagePathList.size(); j++)
        {
            int radius, neighbors;
            //假设
            radius = 1;         //半径越小 图像越清晰 精细 
            neighbors = 8;      //领域数目越小，图像亮度越低，合理，4太小了比较黑 设置8比较合理

            //读取图片
            auto img = imread(imagePathList[j]);

            //标签图像第n行的首地址
            int* labelPtr = label.ptr<int>(n);
            //标签图像赋值
            labelPtr[0] = i;

            Mat lbpImg = Mat(imgRows, imgCols, CV_8UC1, Scalar(0));
            //转换成灰度图
            cvtColor(img, img, COLOR_BGR2GRAY);
            //调整大小，便于经过圆形LBP计算后分割，这里将原始图像放大
            Mat shrink;
            resize(img, shrink, Size(imgCols + 2 * radius, imgRows + 2 * radius), 0, 0, CV_INTER_LINEAR);

            //提取lbp特征
            elbp(shrink, lbpImg, 1, 8);
            //提取特征向量
            Mat m = getLBPH(lbpImg, lbpNum, 17, 4, false);
            m.row(0).copyTo(inputImg.row(n));
            n++;

/*            imshow("img", lbpImg);
            waitKey(0);      */     
        }
    }
}

//获取图片数量
int getNum(const string path)
{
    int num = 0;
    for (int i = 0; i < 2; i++)
    {
        char folder[100];

        if (i == 0)
        {
            //负样本
            sprintf_s(folder, "%s/%s", path.c_str(), "no");
        }
        else
        {
            //正样本
            sprintf_s(folder, "%s/%s", path.c_str(), "has");
        }
        vector<cv::String> imagePathList;
        //读取路径下所有图片
        glob(folder, imagePathList);

        num = num + imagePathList.size();

    }
    return num;
}




//随机打乱训练集和标签
void RandomArray(Mat Train, Mat Label, int num)
{
    int tmp;
    Mat img;

    srand((int)time(NULL));
    for (int i = 0; i < num; i++)
    {
        tmp = rand() % num;

        Train.row(i).copyTo(img);
        Train.row(tmp).copyTo(Train.row(i));
        img.copyTo(Train.row(tmp));

        int t2 = Label.at<int>(i, 0);
        Label.at<int>(i, 0) = Label.at<int>(tmp, 0);
        Label.at<int>(tmp, 0) = t2;
    }
}






