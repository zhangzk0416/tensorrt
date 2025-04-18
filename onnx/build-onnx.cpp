
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

inline const char* severity_string(nvinfer1::ILogger::Severity t) {
	switch (t) {
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR:   return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO:    return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknow";
	}
}

class TRTLogger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kINFO) {
			// 打印带颜色的字符
			// 其中\033[是起始标记
			// 47		是背景颜色
			// ;		分隔符
			// 33		文字颜色
			// m        开始标记结束
			if (severity == Severity::kWARNING) {
				printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else if (severity <= Severity::kERROR) {
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else {
				// 通过severity_string将日志等级数字转为对应的字符串
				printf("%s: %s\n", severity_string(severity), msg);
			}
		}
	}
} logger;

bool build_model() {
	TRTLogger logger;

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);


	// ----------------------------- 2. 输入，模型结构和输出的基本信息-----------------------------
	// 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	if (!parser->parseFromFile("F:/and/demo.onnx", 1)) {
		printf("Failed to parser demo.onnx\n");

		// 这里有几个指针需要释放
		return false;
	}

	int maxBatchSize = 10;
	printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);

	// --------------------------------- 2.1 profile ----------------------------------
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	int input_channel = input_tensor->getDimensions().d[1];

	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
	
	config->addOptimizationProfile(profile);

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	if (engine == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	
	nvinfer1::IHostMemory* model_data = engine->serialize();
	std::ofstream outfile("F:/and/mlp.engine", std::ios::binary);
	cout << model_data->size() << endl;
	outfile.write((char *)model_data->data(), model_data->size());
	outfile.close();

	model_data->destroy();
	parser->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
	printf("Done.\n");
	return true;
}

int main() {
	build_model();
	return 0;
}
