#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui_c.h>
#include <fstream>

// ģ�������ļ�prototxt
std::string model_txt_file = "/home/orangepi/Desktop/caffeDNNtest/src/bvlc_googlenet.prototxt";
// ������caffeģ���ļ�
std::string model_bin_file = "/home/orangepi/Desktop/caffeDNNtest/src/bvlc_googlenet.caffemodel";
// ģ��label��ǩ�ļ�
std::string labels_txt_file = "/home/orangepi/Desktop/caffeDNNtest/src/synset_words.txt";

std::vector<std::string> readLabels();	// ��ȡ��ǩ�ļ�

/*
 * ������
 */
int main(int argc, char** argv) {
	cv::Mat src = cv::imread("/home/orangepi/Desktop/caffeDNNtest/src/a.jpg");	// ��ȡͼƬ
	if (src.empty()) {					// ���ݷǿ�
		printf("Could not load image...\n");
		return -1;
	}
	//cv::namedWindow("input", CV_WINDOW_AUTOSIZE);
	//cv::imshow("input", src);			// ��ʾԭͼ��

	std::vector<std::string> labels = readLabels();		// ��ȡ�洢��ǩ�ļ�
	// ����caffeģ���ļ�������ģ�������ļ���������ģ��caffemodel
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(model_txt_file, model_bin_file);
	if (net.empty()) {					// ģ���ļ��ǿ�
		printf("read caffe model data failure...\n");
		return -1;
	}
	// ת������ͼƬ�ļ���ʽ�����ű������ߴ磬��һ��
	cv::Mat intputBlob = cv::dnn::blobFromImage(src, 1.0, cv::Size(224, 224), cv::Scalar(104, 117, 123));
	cv::Mat prob;		// 
	for (int i = 0; i < 10; i++) {			// Ϊɶѭ��������
		net.setInput(intputBlob, "data");	// �����������ݣ�������ģ�������ļ�����һ��
		prob = net.forward("prob");			// ����������ƣ���ģ�������ļ�����һ��
	}
	cv::Mat probMat = prob.reshape(1, 1);	// reshape��1��1000��
	cv::Point classNumber;					// �洢���λ�õ�����
	double classProb;						// �洢��������
	// Ѱ����������У����ֵ����Сֵ��λ�ú���ֵ
	cv::minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;	// �������ֵ��X����
	printf("\ncurrent image classification : %s, possible : %.2f\n", labels.at(classidx).c_str(), classProb);
	//putText(src, labels.at(classidx), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);

	//cv::imshow("Image Classification", src);
	//cv::waitKey(0);
	return 0;
}



/* 
 * ��ȡ��ǩ�ļ� 
 */
std::vector<std::string> readLabels() {
	std::vector<std::string> classNames;	// �洢�����������
	std::ifstream fp(labels_txt_file);		// �򿪱�ǩ�ļ�
	if (!fp.is_open()) {					// ���û�д�
		printf("Could not open the file");
		exit(-1);
	}
	std::string name;			// �洢һ��������Ƶ�����
	while (!fp.eof()) {			// û�еĶ�ȡ�����һ�оͼ�����ȡ
		std::getline(fp, name);	// ��ȡһ�У��洢������
		if (name.length()) {	// ���������
			// �ҵ�һ���пո񣬵������1���ַ����洢�������ĺ���
			classNames.push_back(name.substr(name.find(' ') + 1));
		}
	}
	fp.close();
	return classNames;
}






