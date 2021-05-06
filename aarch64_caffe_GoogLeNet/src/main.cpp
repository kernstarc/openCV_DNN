#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui_c.h>
#include <fstream>

// 模型描述文件prototxt
std::string model_txt_file = "/home/orangepi/Desktop/caffeDNNtest/src/bvlc_googlenet.prototxt";
// 二进制caffe模型文件
std::string model_bin_file = "/home/orangepi/Desktop/caffeDNNtest/src/bvlc_googlenet.caffemodel";
// 模型label标签文件
std::string labels_txt_file = "/home/orangepi/Desktop/caffeDNNtest/src/synset_words.txt";

std::vector<std::string> readLabels();	// 读取标签文件

/*
 * 主函数
 */
int main(int argc, char** argv) {
	cv::Mat src = cv::imread("/home/orangepi/Desktop/caffeDNNtest/src/a.jpg");	// 读取图片
	if (src.empty()) {					// 数据非空
		printf("Could not load image...\n");
		return -1;
	}
	//cv::namedWindow("input", CV_WINDOW_AUTOSIZE);
	//cv::imshow("input", src);			// 显示原图像

	std::vector<std::string> labels = readLabels();		// 读取存储标签文件
	// 加载caffe模型文件，输入模型描述文件，二进制模型caffemodel
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(model_txt_file, model_bin_file);
	if (net.empty()) {					// 模型文件非空
		printf("read caffe model data failure...\n");
		return -1;
	}
	// 转换输入图片文件格式，缩放比例，尺寸，归一化
	cv::Mat intputBlob = cv::dnn::blobFromImage(src, 1.0, cv::Size(224, 224), cv::Scalar(104, 117, 123));
	cv::Mat prob;		// 
	for (int i = 0; i < 10; i++) {			// 为啥循环做推理
		net.setInput(intputBlob, "data");	// 设置输入数据，名称与模型描述文件输入一致
		prob = net.forward("prob");			// 设置输出名称，与模型描述文件输入一致
	}
	cv::Mat probMat = prob.reshape(1, 1);	// reshape成1行1000列
	cv::Point classNumber;					// 存储类别位置的坐标
	double classProb;						// 存储类别的名称
	// 寻找类别向量中，最大值，最小值的位置和数值
	cv::minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;	// 保存最大值的X坐标
	printf("\ncurrent image classification : %s, possible : %.2f\n", labels.at(classidx).c_str(), classProb);
	//putText(src, labels.at(classidx), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);

	//cv::imshow("Image Classification", src);
	//cv::waitKey(0);
	return 0;
}



/* 
 * 读取标签文件 
 */
std::vector<std::string> readLabels() {
	std::vector<std::string> classNames;	// 存储所有类别名称
	std::ifstream fp(labels_txt_file);		// 打开标签文件
	if (!fp.is_open()) {					// 如果没有打开
		printf("Could not open the file");
		exit(-1);
	}
	std::string name;			// 存储一行类别名称的内容
	while (!fp.eof()) {			// 没有的读取到最后一行就继续读取
		std::getline(fp, name);	// 读取一行，存储到变量
		if (name.length()) {	// 如果有数据
			// 找到一行中空格，到后面的1个字符，存储到向量的后面
			classNames.push_back(name.substr(name.find(' ') + 1));
		}
	}
	fp.close();
	return classNames;
}






