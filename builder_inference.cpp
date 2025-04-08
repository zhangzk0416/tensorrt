#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"

#include <cuda_runtime.h>
#include<assert.h>

#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;


class TRTLogger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kINFO) {
			printf("%d: %s\n", severity, msg);
		}
	}
} ;

// 假设这个函数是在某个命名空间或全局作用域中定义的  const float* values可以不使用const限定符  
nvinfer1::Weights make_weights(const float* values, int count) {
	nvinfer1::Weights weights;
	weights.type = nvinfer1::DataType::kFLOAT; // 指定数据类型为浮点数
	weights.values = const_cast<float*>(values); // 移除 const 限定符，因为 TensorRT API 需要非 const 指针
	weights.count = count; // 设置权重值的数量

	// 注意：这里我们没有为权重值分配新的内存。相反，我们假设 values 指针指向的是已经存在的、有效的内存区域。
	// 调用者需要确保这块内存区域在 TensorRT 引擎使用这些权重值之前和期间都是有效的。

	// 如果权重值是通过 new 或类似方式动态分配的，调用者需要在适当的时候释放这块内存。
	// 如果权重值是从一个 std::vector 或类似容器中获取的，那么容器将负责管理其生命周期。

	return weights;
}

bool build_model() {
	/*生成引擎的过程
		1. 创建builder
		2. 创建网络定义：builder-->network
			构建网络
				创建输入信息
				输入指定数据的名称、数据类型和完整维度，并将输入层添加到网络
				添加各层的权重信息并添加到网络
				标记输出层
		3. 配置参数：builder-->config
		4. 生成engine：builder-->engine()
		5. 序列化保存:engine-->serialize
		6. 释放资源：delete
	*/

	TRTLogger logger;

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 创建network，1表示显性batch size，0表示隐性batch size，目前都采用显性batch size
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	/*构建一个模型
		Network definition:
		image
		  |
		linear  input=3，output=2，bias=true
		  |
		relu
		  |
		prob
	*/
	// 创建输入的信息
	const int num_input = 3;
	const int num_output = 2;
	float layer1_weight_values[] = { 1.0, 2.0, 0.5, 0.1, 0.2, 0.5 };
	float layer1_bias_values[] = { 0.3, 0.8 };

	// 输入指定数据的名称、数据类型和完整维度，并将输入层添加到网络
	nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));

	// 添加layer1层的权重
	nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
	nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 2);
	// 添加全连接层到网络
	auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
	//添加激活层
	auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

	// 将prob标记为输出层
	network->markOutput(*prob->getOutput(0));

	printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);
	builder->setMaxBatchSize(1);

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	if (engine == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	nvinfer1::IHostMemory* model_data = engine->serialize();
	std::ofstream outfile("F:/and/mlp.engine", std::ios::binary);
	cout << model_data->size() << endl;
	outfile.write((char *)model_data->data(), model_data->size());

	model_data->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
	printf("Done.\n");
	return true;
}
vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}


void inference() {

	/*使用cu文件时希望使用cuda的编译器，会自动链接cuda库
	runtime推理过程
		创建logger
		1. 创建一个runtime对象
		读取引擎数据
		2. 反序列化生成engine：runtime-->engine
		3. 创建一个执行上下文ExecutionContext：engine-->context
		4. 填充数据
			准备输入
			创建变量
			分配空间
			将数据放到gpu
		5. 执行推理
		6. 释放资源
	*/

	TRTLogger logger;
	auto engine_data = load_file("F:/and/mlp.engine");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if (engine == nullptr) {
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return;
	}

	nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
	cudaStream_t stream = nullptr;
	// 创建cuda流，以确定这个batch的独立运行
	cudaStreamCreate(&stream);


	// 准备输入并放到gpu上
	float input_data_host[] = { 1, 2, 3 };
	float* input_data_device = nullptr;

	float output_data_host[2];
	float* output_data_device = nullptr;
	cudaMalloc(&input_data_device, sizeof(input_data_host));
	cudaMalloc(&output_data_device, sizeof(output_data_host));
	cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);
	// 用一个指针数组指定input和output在gpu中的指针
	float* bindings[] = { input_data_device, output_data_device };

	// 推理并将结果放回cpu
	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
	cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
	//同步cuda流的异步操作，确保拷贝完成
	cudaStreamSynchronize(stream);

	printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);
	cudaStreamDestroy(stream);
	execution_context->destroy();
	engine->destroy();
	runtime->destroy();

	const int num_input = 3;
	const int num_output = 2;
	float layer1_weight_values[] = { 1.0, 2.0, 0.5, 0.1, 0.2, 0.5 };
	float layer1_bias_values[] = { 0.3, 0.8 };

	for (int io = 0; io < num_output; ++io) {
		float output_host = layer1_bias_values[io];
		for (int ii = 0; ii < num_input; ++ii) {
			output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
		}

		// sigmoid
		float prob = 1 / (1 + exp(-output_host));
		printf("output_prob[%d] = %f\n", io, prob);
	}
}

int main() {

	if (!build_model()) {
		return -1;
	}
	inference();
	return 0;
}
