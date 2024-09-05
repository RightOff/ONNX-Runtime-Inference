#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <thread>
#include <functional>


// 模型配置信息
struct Model_Configuration
{
	public: 
	float confThreshold;// Confidence threshold置信度阈值
	float nmsThreshold;  // Non-maximum suppression threshold非最大抑制阈值
	float objThreshold;  //Object Confidence threshold对象置信度阈值
	std::string modelpath;
    int inpWidth;   //输入图像宽度
	int inpHeight;  //输入图像长度
    int num_classes;    //类别数
	const int64_t NumThreads = 4;	//线程数
	bool useCUDA;	//使用CUDA加速
};

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;


class YOLOv5{
public:
    YOLOv5(Model_Configuration yolov5_config);  //YOLOv5模型初始化
    void detect(cv::Mat& frame);    //输入图片进行检测
    
private:
    float confThreshold;    // Confidence threshold置信度阈值
	float nmsThreshold;     // NMS阈值
	float objThreshold;     //Object Confidence threshold对象置信度阈值
	int64_t NumThreads;	//线程数
	bool useCUDA;	//使用CUDA加速

	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	int num_classes;
    const bool keep_ratio = true;
	std::vector<float> input_image_;		// 输入图片
	//类别信息
	std::string classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
							"train", "truck", "boat", "traffic light", "fire hydrant",
							"stop sign", "parking meter", "bench", "bird", "cat", "dog",
							"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
							"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
							"skis", "snowboard", "sports ball", "kite", "baseball bat",
							"baseball glove", "skateboard", "surfboard", "tennis racket",
							"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
							"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
							"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
							"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
							"remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
							"sink", "refrigerator", "book", "clock", "vase", "scissors",
							"teddy bear", "hair drier", "toothbrush"};

    
    cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left);    //修改图片大小并填充边界防止失真
	void normalize_(cv::Mat img);	//归一化
	void nms(std::vector<BoxInfo>& input_boxes);	//NMS
    
    Ort::Session *ort_session = nullptr;    // 初始化Session指针选项
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();  //创建Session选项
    std::string instanceName{"tired_detection"};
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, instanceName.c_str()); // 初始化环境

	//SessionOptions sessionOptions;
    std::vector<char*> input_names;  // 定义一个字符指针vector
	std::vector<char*> output_names; // 定义一个字符指针vector
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs  ，二维vector
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++标准

};

//修改图片大小并填充边界防止失真
cv::Mat YOLOv5::resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left){
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
	*neww = this->inpWidth;
    cv::Mat dstimg;

	//不太理解逻辑？？
    if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {		
			// 如果高度大于宽度，调整宽度保持比例
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			// 左右填充以保持比例
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else {
			// 如果宽度大于高度，调整高度保持比例
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);  //等比例缩小，防止失真
			// 上下填充以保持比例
			*top = (int)(this->inpHeight - *newh) * 0.5;  //上部缺失部分
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114); //上部填补top大小，下部填补剩余部分，左右不填补
		}
	}
	else {
		resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}
//归一化
void YOLOv5::normalize_(cv::Mat img)  {
	//    img.convertTo(img, CV_32F);
    //cout<<"picture size"<<img.rows<<img.cols<<img.channels()<<endl;
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector大小

	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // 行
		{
			for (int j = 0; j < col; j++)  // 列
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
				this->input_image_[c * row * col + i * col + j] = pix / 255.0; //将每个像素块归一化后装进容器
			}
		}
	}
}

void YOLOv5::nms(std::vector<BoxInfo>& input_boxes){
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	std::vector<bool> remove_flags(input_boxes.size(),false);
	auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	{
		float xx1 = std::max(box1.x1, box2.x1);
		float yy1 = std::max(box1.y1, box2.y1);
		float xx2 = std::min(box1.x2, box2.x2);
		float yy2 = std::min(box1.y2, box2.y2);
		// 交集
		float w = std::max(0.0f, xx2 - xx1 + 1);
		float h = std::max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
		float union_area = std::max(0.0f,box1.x2-box1.x1) * std::max(0.0f,box1.y2-box1.y1)
						   + std::max(0.0f,box2.x2-box2.x1) * std::max(0.0f,box2.y2-box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if(remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if(remove_flags[j]) continue;
			if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->nmsThreshold)
			{
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}

//YOLOv5模型初始化
YOLOv5::YOLOv5(Model_Configuration config){
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->useCUDA = config.useCUDA;
	this->NumThreads = config.NumThreads;
	
    this->num_classes = 80;//sizeof(this->classes)/sizeof(this->classes[0]);  // 类别数量
	this->inpHeight = 640;
	this->inpWidth = 640;
    std::string model_path = config.modelpath;
	
	if (this->useCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        // OrtTensorRTProviderOptions trt_options{};
        // sessionOptions.AppendExecutionProvider_TensorRT(trt_options)
    }
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //设置图优化类型
	sessionOptions.SetIntraOpNumThreads(this->NumThreads);	//好像并不能加速推理，意义何在？？
    ort_session = new Ort::Session(env, (const char*)model_path.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount();  //输入输出节点数量                         
	size_t numOutputNodes = ort_session->GetOutputCount(); 
	Ort::AllocatorWithDefaultOptions allocator;   // 配置输入输出节点内存

    for (int i = 0; i < numInputNodes; i++)
	{
        // 将输出名称存储在vector中
		input_names.push_back(ort_session->GetInputName(i, allocator));		// 内存
        // 获取类型信息
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  // 
		auto input_dims = input_tensor_info.GetShape();    // 输入shape
		input_node_dims.push_back(input_dims);	// 保存
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];      // 5+classes
	this->num_proposal = output_node_dims[0][1];  // pre_box

}
void RunInference(
    Ort::Session* session,
    const std::vector<char*>& input_names,
    Ort::Value* input_tensor,
    const std::vector<char*>& output_names)
{
    // 调用 ONNX Runtime 的 Run 方法
    session->Run(Ort::RunOptions{ nullptr }, &input_names[0], 
                        input_tensor, 1, output_names.data(), output_names.size());
	std::cout << "Thread ID: " << std::this_thread::get_id() << std::endl;
}
void YOLOv5::detect(cv::Mat& frame){
    int newh = 0, neww = 0, padh = 0, padw = 0;
    cv::Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);   //改大小后做padding防失真
	this->normalize_(dstimg);       //归一化
	// 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
	std::array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };  //1,3,640,640

	//创建输入tensor
    /*
    这一行代码的作用是创建一个指向CPU内存的分配器信息对象(AllocatorInfo)，用于在运行时分配和释放CPU内存。
    它调用了CreateCpu函数并传递两个参数：OrtDeviceAllocator和OrtMemTypeCPU。
    其中，OrtDeviceAllocator表示使用默认的设备分配器，OrtMemTypeCPU表示在CPU上分配内存。
    通过这个对象，我们可以在运行时为张量分配内存，并且可以保证这些内存在计算完成后被正确地释放，避免内存泄漏的问题。
    */
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	//使用Ort库创建一个输入张量，其中包含了需要进行目标检测的图像数据。
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), 
		input_image_.size(), input_shape_.data(), input_shape_.size());
	// 开始推理
	std::vector<Ort::Value> ort_outputs = ort_session->Run(Ort::RunOptions{ nullptr }, &input_names[0], 
		&input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理

	// //多线程处理
	// using RunType = std::vector<Ort::Value>(Ort::Session::*)(const Ort::RunOptions&, const char* const*, const Ort::Value*, size_t, const char* const*, size_t);
	// std::thread th1(std::bind(static_cast<RunType>(&Ort::Session::Run), ort_session, Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size()));
	// th1.join();
	std::thread thread1(RunInference, ort_session, std::ref(input_names), &input_tensor_, std::ref(output_names));
	std::thread thread2(RunInference, ort_session, std::ref(input_names), &input_tensor_, std::ref(output_names));
	std::thread thread3(RunInference, ort_session, std::ref(input_names), &input_tensor_, std::ref(output_names));
	std::thread thread4(RunInference, ort_session, std::ref(input_names), &input_tensor_, std::ref(output_names));
	thread1.join();
	thread2.join();
	thread3.join();
	thread4.join();
	
	std::vector<BoxInfo> generate_boxes;  // BoxInfo自定义的结构体
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;  //原图高和新高比，原图宽与新宽比
	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData
	
	for(int i = 0; i < num_proposal; ++i) // 遍历所有的num_pre_boxes
	{   
		int index = i * nout;      // prob[b*num_pred_boxes*(classes+5)]  
		float obj_conf = pdata[index + 4];  // 置信度分数
        //cout<<"k"<<obj_conf<<endl;
		if (obj_conf > this->objThreshold)  // 大于阈值
		{
			int class_idx = 0;
			float max_class_socre = 0;
			for (int k = 0; k < this->num_classes; ++k)
			{
				
				if (pdata[k + index + 5] > max_class_socre)
				{
					
					max_class_socre = pdata[k + index + 5];
					class_idx = k;
					
				}
				
				
			}
			max_class_socre *= obj_conf;   // 最大的类别分数*置信度
			if (max_class_socre > this->confThreshold) // 再次筛选
			{ 
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index];  //x
				float cy = pdata[index+1];  //y
				float w = pdata[index+2];  //w
				float h = pdata[index+3];  //h
				//cout<<cx<<cy<<w<<h<<endl;
				float xmin = (cx - padw - 0.5 * w)*ratiow;
				float ymin = (cy - padh - 0.5 * h)*ratioh;
				float xmax = (cx - padw + 0.5 * w)*ratiow;
				float ymax = (cy - padh + 0.5 * h)*ratioh;
				//cout<<xmin<<ymin<<xmax<<ymax<<endl;
				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	
	this->nms(generate_boxes);
	
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(frame, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), cv::Scalar(0, 0, 255), 2);
		std::string label = cv::format("%.2f", generate_boxes[i].score);
		label = this->classes[generate_boxes[i].label] + ":" + label;
		putText(frame, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
	}

}


int main(int argc,char *argv[]){

	bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
	const int64_t NumThreads = 4;
    if (argc == 1)
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) != 0))
    {
        useCUDA = false;
    }
    else
    {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useCUDA)
    {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }


    std::string imageFilepath{
        "../../data/images/european-bee-eater-2115564_1920.jpg"};	//测试图片位置

    std::string instanceName{"tired_detection"};
	
    Model_Configuration yolov5_config = { 0.3, 0.5, 0.3,"../../data/models/yolov5n.onnx",80,640,640,NumThreads,useCUDA};  //设置模型配置信息
    YOLOv5 yolov5_model(yolov5_config);	//创建检测实例

	cv::namedWindow("tired_detect", cv::WINDOW_NORMAL);
    while(true){

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		
		cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

		yolov5_model.detect(imageBGR);

		
		cv::imshow("tired_detect",imageBGR);

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    	std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
              << " ms" << std::endl;


		if (cv::waitKey(1) == ' '){//按键采集，用户按下' ',跳出循环,结束采集
			std::cout << "----------------------------------" << std::endl;
			std::cout << "------------- closed -------------" << std::endl;
			std::cout << "----------------------------------" << std::endl;
			break;
		}
		
    }
	// std::cout << "Press Enter to continue...";
    // std::cin.ignore();
	cv::destroyAllWindows();
	return 0;
}
