#include "RK35llm.h"

#define HISTORY true
//----------------------------------------------------------------------------------------
RK35llm::RK35llm(void)
{
    Info     = false;
    Silence  = false;
    Thinking = false;
    is_video = false;
    ImgVec   = nullptr;
    llmHandle = nullptr;
    responseReady_ = false;
    
    memset(&rknn_app_ctx, 0, sizeof(RKLLM_app_context_t));
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    memset(&rkllm_cb, 0, sizeof(RKLLMCallback)); 

    param = rkllm_createDefaultParam();
    param.top_k = 1;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;
}
//----------------------------------------------------------------------------------------
RK35llm::~RK35llm(void)
{
    if(rknn_app_ctx.input_attrs != nullptr){
        free(rknn_app_ctx.input_attrs);
        rknn_app_ctx.input_attrs = nullptr;
    }
    if(rknn_app_ctx.output_attrs != nullptr){
        free(rknn_app_ctx.output_attrs);
        rknn_app_ctx.output_attrs = nullptr;
    }
    if(rknn_app_ctx.rknn_ctx != 0){
        rknn_destroy(rknn_app_ctx.rknn_ctx);
        rknn_app_ctx.rknn_ctx = 0;
    }
    if(ImgVec!=nullptr){
        delete[] ImgVec;
        ImgVec = nullptr;
    }
    if(llmHandle!=nullptr) rkllm_destroy(llmHandle);
}
//----------------------------------------------------------------------------------------
void RK35llm::DumpTensorAttr(rknn_tensor_attr* attr)
{
    if(!Info) return;
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
//----------------------------------------------------------------------------------------
int RK35llm::StaticCallback(RKLLMResult* result, void* userdata, LLMCallState state)
{
    if (!userdata) return -1;
    RK35llm* self = static_cast<RK35llm*>(userdata);
    return self->InstanceCallback(result, state);
}
//----------------------------------------------------------------------------------------
int RK35llm::InstanceCallback(RKLLMResult *result, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH) {
        if(!Silence) printf("\n");
        responseReady_ = true;
        responseCv_.notify_all();
    }
    else if (state == RKLLM_RUN_ERROR) {
        if(!Silence) printf("[Error during inference]\n");
        responseBuffer_ += "[Error during inference]";
        responseReady_ = true;
        responseCv_.notify_all();
    }
    else if (state == RKLLM_RUN_NORMAL) {
        if(!Silence) printf("%s", result->text);
        if (result && result->text) responseBuffer_ += result->text;
    }
    return 0;
}
//----------------------------------------------------------------------------------------
cv::Mat RK35llm::Expand2Square(const cv::Mat& img, const cv::Scalar& background_color)
{
    int width = img.cols;
    int height = img.rows;
    if (width == height) return img.clone();

    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));
    return result;
}
//----------------------------------------------------------------------------------------
int RK35llm::InitImgEnc(const char* model_path)
{
    int ret;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (void*)model_path, 0, 0, NULL);
    if (ret < 0) return -1;

    int core_num=3;
    if (core_num == 2) rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
    else if (core_num == 3) rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    else rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return -1;

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }
    
    for (int i = 0; i < 4; i++) {
        if (output_attrs[0].dims[i] > 1) {
            rknn_app_ctx.model_image_token = output_attrs[0].dims[i];
            rknn_app_ctx.model_embed_size = output_attrs[0].dims[i + 1];
            break;
        }
    }
    
    rknn_app_ctx.rknn_ctx = ctx;
    rknn_app_ctx.io_num = io_num;
    rknn_app_ctx.input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx.output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        rknn_app_ctx.model_channel = input_attrs[0].dims[1];
        rknn_app_ctx.model_height  = input_attrs[0].dims[2];
        rknn_app_ctx.model_width   = input_attrs[0].dims[3];
    } else {
        rknn_app_ctx.model_height  = input_attrs[0].dims[1];
        rknn_app_ctx.model_width   = input_attrs[0].dims[2];
        rknn_app_ctx.model_channel = input_attrs[0].dims[3];
    }
    return 0;
}
//----------------------------------------------------------------------------------------
int RK35llm::RunImgEnc(float* dest_ptr) // NEW: Accepts dest pointer for contiguous memory
{
    int ret = 0;
    rknn_input inputs[1];
    rknn_output outputs[rknn_app_ctx.io_num.n_output];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
    inputs[0].buf   = resized_img.data;

    ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, 1, inputs);
    if (ret < 0) return -1;

    ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
    if (ret < 0) return -1;

    for (uint32_t j=0; j<rknn_app_ctx.io_num.n_output; j++) {
        outputs[j].want_float = 1;
    }
    ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
    if (ret < 0) return ret;

    if(rknn_app_ctx.io_num.n_output == 1) memcpy(dest_ptr, outputs[0].buf, outputs[0].size);
    else {
        for(int i=0; i<rknn_app_ctx.model_image_token; i++){
            for (uint32_t j = 0; j < rknn_app_ctx.io_num.n_output; j++) {
                memcpy(dest_ptr + i * rknn_app_ctx.io_num.n_output * rknn_app_ctx.model_embed_size + j * rknn_app_ctx.model_embed_size,
                      (float*)(outputs[j].buf) + i * rknn_app_ctx.model_embed_size, sizeof(float) * rknn_app_ctx.model_embed_size);
            }
        }
    }

    rknn_outputs_release(rknn_app_ctx.rknn_ctx, 1, outputs);
    return ret;
}
//----------------------------------------------------------------------------------------
void RK35llm::SetInfo(bool _Info) { Info = _Info; }
void RK35llm::SetSilence(bool _Silence) { Silence = _Silence; }
void RK35llm::SetThinking(bool _Thinking) { Thinking = _Thinking; }
//----------------------------------------------------------------------------------------
void RK35llm::SetHistory(bool _History)
{
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;
    if(_History){
        rkllm_infer_params.keep_history = 1;
        rkllm_set_chat_template(llmHandle, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");
    }
}
//----------------------------------------------------------------------------------------
bool RK35llm::LoadModel(const std::string& VLMmodel, const std::string& LLMmodel, int32_t NewTokens, int32_t ContextLength)
{
    param.model_path = LLMmodel.c_str();
    param.max_new_tokens = NewTokens;
    param.max_context_len = ContextLength;
    param.n_keep = (ContextLength > 2048) ? 2048 : ContextLength / 2;

    rkllm_cb.result_callback = RK35llm::StaticCallback;
    rkllm_cb.result_userdata = this;

    int ret = rkllm_init(&llmHandle, &param, &rkllm_cb);
    if(ret != 0) return false;
    else if(Info) printf("rkllm init success\n");

    #if HISTORY
        rkllm_infer_params.keep_history = 1;
        rkllm_set_chat_template(llmHandle,
             "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
             "<|im_start|>user\n",
             "<|im_end|>\n<|im_start|>assistant\n");
    #else
        rkllm_infer_params.keep_history = 0;
    #endif

    ret = InitImgEnc(VLMmodel.c_str());
    if(ret != 0) return false;

    return true;
}
//----------------------------------------------------------------------------------------
void RK35llm::LoadImage(const cv::Mat& img)
{
    if (llmHandle) rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
    
    is_video = false;
    current_img_count = 1;
    
    // Dynamic memory allocation for 1 frame
    size_t vec_size = rknn_app_ctx.model_image_token * rknn_app_ctx.model_embed_size * rknn_app_ctx.io_num.n_output;
    if (ImgVec != nullptr) delete[] ImgVec;
    ImgVec = new float[vec_size];
    memset(ImgVec, 0, vec_size * sizeof(float));

    cv::Mat rgb_img = img.clone();
    cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);

    cv::Scalar background_color(127.5, 127.5, 127.5);
    cv::Mat square_img = Expand2Square(rgb_img, background_color);

    cv::Size new_size(rknn_app_ctx.model_width, rknn_app_ctx.model_height);
    cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

    RunImgEnc(ImgVec); // Pass base pointer
}
//----------------------------------------------------------------------------------------
void RK35llm::LoadVideoFrames(const std::vector<cv::Mat>& frames)
{
    if (frames.empty()) return;
    
    if (llmHandle) rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
    
    is_video = true;
    current_img_count = frames.size();
    
    // Dynamic memory allocation for N consecutive frames
    size_t single_frame_size = rknn_app_ctx.model_image_token * rknn_app_ctx.model_embed_size * rknn_app_ctx.io_num.n_output;
    size_t total_vec_size = single_frame_size * current_img_count;
    
    if (ImgVec != nullptr) delete[] ImgVec;
    ImgVec = new float[total_vec_size];
    memset(ImgVec, 0, total_vec_size * sizeof(float));

    cv::Scalar background_color(127.5, 127.5, 127.5);
    
    for (int i = 0; i < current_img_count; ++i) {
        cv::Mat rgb_img = frames[i].clone();
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);

        cv::Mat square_img = Expand2Square(rgb_img, background_color);
        cv::Size new_size(rknn_app_ctx.model_width, rknn_app_ctx.model_height);
        cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

        // Run NPU and write output directly to the correct offset in the contiguous block
        RunImgEnc(ImgVec + (i * single_frame_size));
    }
}
//----------------------------------------------------------------------------------------
std::string RK35llm::Ask(const std::string& Question)
{
    std::string Str="";
    if (!llmHandle) return Str;

    {
        std::lock_guard<std::mutex> lk(responseMutex_);
        responseBuffer_.clear();
        responseReady_ = false;
    }

    if (Question == "clear") {
        rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
        current_img_count = 0;
        return Str;
    }

    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    rkllm_input.enable_thinking = Thinking;

    // Detect multimodal intent (checks for both image and video tags)
    if (Question.find("<image>") == std::string::npos && Question.find("<video>") == std::string::npos) {
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = Question.c_str();
    } else if (is_video) {
        // Video Struct Population
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.role = "user";
        rkllm_input.multimodal_input.prompt = (char*)Question.c_str();
        
        rkllm_input.multimodal_input.video.video_embed = ImgVec;
        rkllm_input.multimodal_input.video.n_frame_tokens = rknn_app_ctx.model_image_token;
        rkllm_input.multimodal_input.video.n_frame_per_video = current_img_count;
        rkllm_input.multimodal_input.video.n_video = 1;
        rkllm_input.multimodal_input.video.frame_height = rknn_app_ctx.model_height;
        rkllm_input.multimodal_input.video.frame_width = rknn_app_ctx.model_width;

        rkllm_input.multimodal_input.video.video_start = "<|video_start|>";
        rkllm_input.multimodal_input.video.video_end = "<|video_end|>";
        rkllm_input.multimodal_input.video.video_content = "<|video_pad|>";
    } else {
        // Image Struct Population
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.role = "user";
        rkllm_input.multimodal_input.prompt = (char*)Question.c_str();
        
        rkllm_input.multimodal_input.image.image_embed = ImgVec;
        rkllm_input.multimodal_input.image.n_image_tokens = rknn_app_ctx.model_image_token;
        rkllm_input.multimodal_input.image.n_image = 1;
        rkllm_input.multimodal_input.image.image_height = rknn_app_ctx.model_height;
        rkllm_input.multimodal_input.image.image_width = rknn_app_ctx.model_width;

        rkllm_input.multimodal_input.image.image_start = "<|vision_start|>";
        rkllm_input.multimodal_input.image.image_end = "<|vision_end|>";
        rkllm_input.multimodal_input.image.image_content = "<|image_pad|>";
    }
    
    if(!Silence) printf("Answer: ");

    int ret = rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, this);
    if (ret != 0) std::cerr << "rkllm_run returned " << ret << "\n";

    std::unique_lock<std::mutex> lk(responseMutex_);
    responseCv_.wait(lk, [this]{ return responseReady_; });

    return responseBuffer_;
}
