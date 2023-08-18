#include "CLbp.h"



//2.圆形算子LBP函数
//src:输入图像
//dst:输出图像
//radius:半径
//neighbors:要处理周围像素点的数量
void elbp(Mat& src, Mat& dst, int radius, int neighbors)
{

    for (int n = 0; n < neighbors; n++)
    {
        // 采样点的计算
        float x = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        // 上取整和下取整的值
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // 小数部分
        float ty = y - fy;
        float tx = x - fx;
        // 设置插值权重
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        // 循环处理图像数据
        for (int i = radius; i < src.rows - radius; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                // 计算插值
                float t = static_cast<float>(w1 * src.at<uchar>(i + fy, j + fx) + w2 * src.at<uchar>(i + fy, j + cx) + w3 * src.at<uchar>(i + cy, j + fx) + w4 * src.at<uchar>(i + cy, j + cx));
                // 进行编码
                dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}


//等价模式LBP特征计算
//src:输入图像
//dst:输出图像
//radius:半径
//neighbors:要处理周围像素点的数量
void getUniformPatternLBPFeature(Mat src, Mat dst, int radius, int neighbors)
{
    //LBP特征图像的行数和列数的计算要准确
    dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
    dst.setTo(0);
    //LBP特征值对应图像灰度编码表，直接默认采样点为8位
    uchar temp = 1;
    uchar table[256] = { 0 };
    for (int i = 0; i < 256; i++)
    {
        if (getHopTimes(i) < 3)
        {
            table[i] = temp;
            temp++;
        }
    }
    //是否进行UniformPattern编码的标志
    bool flag = false;
    //计算LBP特征图
    for (int k = 0; k < neighbors; k++)
    {
        if (k == neighbors - 1)
        {
            flag = true;
        }
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));
        //为双线性插值做准备
        //对采样点偏移量分别进行上下取整
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - x1;
        float ty = ry - y1;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        //循环处理每个像素
        for (int i = radius; i < src.rows - radius; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                //获得中心像素点的灰度值
                uint8_t center = src.at<uint8_t>(i, j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = src.at<uint8_t>(i + x1, j + y1) * w1 + src.at<uint8_t>(i + x1, j + y2) * w2 \
                    + src.at<uint8_t>(i + x2, j + y1) * w3 + src.at<uint8_t>(i + x2, j + y2) * w4;
                //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst.at<uchar>(i - radius, j - radius) |= (neighbor > center) << (neighbors - k - 1);
                //进行LBP特征的UniformPattern编码
                if (flag)
                {
                    dst.at<uchar>(i - radius, j - radius) = table[dst.at<uchar>(i - radius, j - radius)];
                }
            }
        }
    }
}
//计算跳变次数
int getHopTimes(int n)
{
    int count = 0;
    bitset<8> binaryCode = n;
    for (int i = 0; i < 8; i++)
    {
        if (binaryCode[i] != binaryCode[(i + 1) % 8])
        {
            count++;
        }
    }
    return count;
}

//计算LBP特征图像的直方图LBPH
//src为LBP是通过lbp计算得到的
//numPatterns为计算LBP的模式数目，一般为2的幂
//grid_x和grid_y分别为每行或每列的block个数
//normed为是否进行归一化处理,1:归一化，0:进行归一化
Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, bool normed)
{
    int width = src.cols / grid_x;
    int height = src.rows / grid_y;
    //定义LBPH的行和列，grid_x*grid_y表示将图像分割成这么些块，numPatterns表示LBP值的模式种类
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    if (src.empty())
    {
        return result.reshape(1, 1);
    }
    int resultRowIndex = 0;
    //对图像进行分割，分割成grid_x*grid_y块，grid_x，grid_y默认为8
    for (int i = 0; i < grid_y; i++)
    {
        for (int j = 0; j < grid_x; j++)
        {
            //图像分块
            Mat src_cell = Mat(src, Range(i * height, (i + 1) * height), Range(j * width, (j + 1) * width));
            //计算直方图
            Mat hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns - 1), normed);
            //将直方图放到result中
            Mat rowResult = result.row(resultRowIndex);
            hist_cell.reshape(1, 1).convertTo(rowResult, CV_32FC1);
            resultRowIndex++;
        }
    }
    return result.reshape(1, 1);
}



//计算一个LBP特征图像块的直方图
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed)
{
    //定义存储直方图的矩阵
    Mat result;
    //计算得到直方图bin的数目，直方图数组的大小
    int histSize = maxValue - minValue + 1;
    //定义直方图每一维的bin的变化范围
    float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) };
    //定义直方图所有bin的变化范围
    const float* ranges = { range };
    //计算直方图，src是要计算直方图的图像，1是要计算直方图的图像数目，0是计算直方图所用的图像的通道序号，从0索引
    //Mat()是要用的掩模，result为输出的直方图，1为输出的直方图的维度，histSize直方图在每一维的变化范围
    //ranges，所有直方图的变化范围（起点和终点）
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &ranges, true, false);
    //归一化
    if (normed)
    {
        result /= (int)src.total();
    }
    //结果表示成只有1行的矩阵
    return result.reshape(1, 1);
}