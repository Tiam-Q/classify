#include "CLbp.h"



//2.Բ������LBP����
//src:����ͼ��
//dst:���ͼ��
//radius:�뾶
//neighbors:Ҫ������Χ���ص������
void elbp(Mat& src, Mat& dst, int radius, int neighbors)
{

    for (int n = 0; n < neighbors; n++)
    {
        // ������ļ���
        float x = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        // ��ȡ������ȡ����ֵ
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // С������
        float ty = y - fy;
        float tx = x - fx;
        // ���ò�ֵȨ��
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        // ѭ������ͼ������
        for (int i = radius; i < src.rows - radius; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                // �����ֵ
                float t = static_cast<float>(w1 * src.at<uchar>(i + fy, j + fx) + w2 * src.at<uchar>(i + fy, j + cx) + w3 * src.at<uchar>(i + cy, j + fx) + w4 * src.at<uchar>(i + cy, j + cx));
                // ���б���
                dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}


//�ȼ�ģʽLBP��������
//src:����ͼ��
//dst:���ͼ��
//radius:�뾶
//neighbors:Ҫ������Χ���ص������
void getUniformPatternLBPFeature(Mat src, Mat dst, int radius, int neighbors)
{
    //LBP����ͼ��������������ļ���Ҫ׼ȷ
    dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
    dst.setTo(0);
    //LBP����ֵ��Ӧͼ��Ҷȱ����ֱ��Ĭ�ϲ�����Ϊ8λ
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
    //�Ƿ����UniformPattern����ı�־
    bool flag = false;
    //����LBP����ͼ
    for (int k = 0; k < neighbors; k++)
    {
        if (k == neighbors - 1)
        {
            flag = true;
        }
        //���������������ĵ������ƫ����rx��ry
        float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));
        //Ϊ˫���Բ�ֵ��׼��
        //�Բ�����ƫ�����ֱ��������ȡ��
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //������ƫ����ӳ�䵽0-1֮��
        float tx = rx - x1;
        float ty = ry - y1;
        //����0-1֮���x��y��Ȩ�ؼ��㹫ʽ����Ȩ�أ�Ȩ�����������λ���޹أ��������Ĳ�ֵ�й�
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        //ѭ������ÿ������
        for (int i = radius; i < src.rows - radius; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                //����������ص�ĻҶ�ֵ
                uint8_t center = src.at<uint8_t>(i, j);
                //����˫���Բ�ֵ��ʽ�����k��������ĻҶ�ֵ
                float neighbor = src.at<uint8_t>(i + x1, j + y1) * w1 + src.at<uint8_t>(i + x1, j + y2) * w2 \
                    + src.at<uint8_t>(i + x2, j + y1) * w3 + src.at<uint8_t>(i + x2, j + y2) * w4;
                //LBP����ͼ���ÿ���ھӵ�LBPֵ�ۼӣ��ۼ�ͨ���������ɣ���Ӧ��LBPֵͨ����λȡ��
                dst.at<uchar>(i - radius, j - radius) |= (neighbor > center) << (neighbors - k - 1);
                //����LBP������UniformPattern����
                if (flag)
                {
                    dst.at<uchar>(i - radius, j - radius) = table[dst.at<uchar>(i - radius, j - radius)];
                }
            }
        }
    }
}
//�����������
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

//����LBP����ͼ���ֱ��ͼLBPH
//srcΪLBP��ͨ��lbp����õ���
//numPatternsΪ����LBP��ģʽ��Ŀ��һ��Ϊ2����
//grid_x��grid_y�ֱ�Ϊÿ�л�ÿ�е�block����
//normedΪ�Ƿ���й�һ������,1:��һ����0:���й�һ��
Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, bool normed)
{
    int width = src.cols / grid_x;
    int height = src.rows / grid_y;
    //����LBPH���к��У�grid_x*grid_y��ʾ��ͼ��ָ����ôЩ�飬numPatterns��ʾLBPֵ��ģʽ����
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    if (src.empty())
    {
        return result.reshape(1, 1);
    }
    int resultRowIndex = 0;
    //��ͼ����зָ�ָ��grid_x*grid_y�飬grid_x��grid_yĬ��Ϊ8
    for (int i = 0; i < grid_y; i++)
    {
        for (int j = 0; j < grid_x; j++)
        {
            //ͼ��ֿ�
            Mat src_cell = Mat(src, Range(i * height, (i + 1) * height), Range(j * width, (j + 1) * width));
            //����ֱ��ͼ
            Mat hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns - 1), normed);
            //��ֱ��ͼ�ŵ�result��
            Mat rowResult = result.row(resultRowIndex);
            hist_cell.reshape(1, 1).convertTo(rowResult, CV_32FC1);
            resultRowIndex++;
        }
    }
    return result.reshape(1, 1);
}



//����һ��LBP����ͼ����ֱ��ͼ
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed)
{
    //����洢ֱ��ͼ�ľ���
    Mat result;
    //����õ�ֱ��ͼbin����Ŀ��ֱ��ͼ����Ĵ�С
    int histSize = maxValue - minValue + 1;
    //����ֱ��ͼÿһά��bin�ı仯��Χ
    float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) };
    //����ֱ��ͼ����bin�ı仯��Χ
    const float* ranges = { range };
    //����ֱ��ͼ��src��Ҫ����ֱ��ͼ��ͼ��1��Ҫ����ֱ��ͼ��ͼ����Ŀ��0�Ǽ���ֱ��ͼ���õ�ͼ���ͨ����ţ���0����
    //Mat()��Ҫ�õ���ģ��resultΪ�����ֱ��ͼ��1Ϊ�����ֱ��ͼ��ά�ȣ�histSizeֱ��ͼ��ÿһά�ı仯��Χ
    //ranges������ֱ��ͼ�ı仯��Χ�������յ㣩
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &ranges, true, false);
    //��һ��
    if (normed)
    {
        result /= (int)src.total();
    }
    //�����ʾ��ֻ��1�еľ���
    return result.reshape(1, 1);
}