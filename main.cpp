
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kVERBOSE){
            printf("%d: %s\n", severity, msg);
        }
    }
};

nvinfer1::Weights make_weights(float* ptr, int n){
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

int main(){
     
    TRTLogger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    const int num_input = 3;   // in_channel
    const int num_output = 2;  // out_channel
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; 
    float layer1_bias_values[]   = {0.3, 0.8};
    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 2);
    //娣诲姞鍏ㄨ繛鎺ュ眰
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);   
    //娣诲姞婵€娲诲眰 
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID); 
    
    // 灏嗘垜浠渶瑕佺殑prob鏍囪涓鸿緭鍑?
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib
    config->setMaxWorkspaceSize(1 << 28);
    builder->setMaxBatchSize(1); // 鎺ㄧ悊鏃?batchSize = 1 

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return -1;
    }


    nvinfer1::IHostMemory* model_data = engine->serialize();


   
    printf("Done.\n");
    return 0;
}