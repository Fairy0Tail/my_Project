#include "detector.h"
#include "test.h"
#include <vector>


int main()
{
	cv::String path = "C:\\Users\\Lenovo\\Desktop\\7";
	//cv::String path = "./images\\";
	std::vector<cv::String> files;
	cv::glob(path, files, false);

	std::vector<cv::Mat> images;
	for (int i = 0; i < files.size(); i++)
	{
		cv::Mat src = cv::imread(files[i]);
		
		images.push_back(src);
	}

	infos info;
	float conf_thres = 0.5;
	std::vector<std::vector<infos>> result_infos;   //���������ÿ��ͼ����ÿ��ȱ�ݵļ����Ϣ
	detect(images, info, result_infos, conf_thres);

	for (int i = 0; i < result_infos.size(); i++)
	{
		std::cout << "\n===========================" << std::endl;
		for (int j = 0; j < result_infos[i].size(); j++)
		{
			std::cout << result_infos[i][j].location << std::endl;  // (���Ͻ�x�����Ͻ�y������)
			std::cout << result_infos[i][j].cls << std::endl;       // ���
			std::cout << result_infos[i][j].conf << std::endl;      // ���Ŷ�
		}
	}

	return 0;
}