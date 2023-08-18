       svm.xml为以训练好的模型可直接使用。
       clbp.cpp和clbp.h文件计算lbp特征向量

       想要做个opencv识别车牌的实战项目，在车牌定位过程中无法百分百准确的筛选出有车牌的区域，于是对图片提取LBP特征向量然后再训练opencv svm分类模型，得到一个分类器。
详情参考：https://blog.csdn.net/ltj5201314/article/details/132367815?spm=1001.2014.3001.5501
模型的正确率在百分之九十九以上。