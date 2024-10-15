#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <random>

// Function to generate random input tensors
std::vector<float> generateRandomInput(const std::vector<int64_t>& dims) {
    size_t total_size = 1;
    for (auto dim : dims) {
        total_size *= dim;
    }
    std::vector<float> data(total_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0, 1);
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

int main() {
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VAEInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Define model paths
    std::string model_dir = "/Users/jihoon/PycharmProjects/PyTorchONNX/onnx_models/";
    std::string vae_full_model = model_dir + "vae_full.onnx";
    std::string encoder_model = model_dir + "vae_encoder.onnx";
    std::string reparameterize_model = model_dir + "vae_reparameterize.onnx";
    std::string decoder_model = model_dir + "vae_decoder.onnx";

    // Create sessions
    Ort::Session vae_full_session(env, vae_full_model.c_str(), session_options);
    Ort::Session encoder_session(env, encoder_model.c_str(), session_options);
    Ort::Session reparameterize_session(env, reparameterize_model.c_str(), session_options);
    Ort::Session decoder_session(env, decoder_model.c_str(), session_options);

    // Create memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Input dimensions
    std::vector<int64_t> input_dims = {1, 1, 28, 28};          // For image inputs
    std::vector<int64_t> latent_dims = {1, 20};                // For latent vectors

    // Generate random inputs
    auto input_data = generateRandomInput(input_dims);
    auto mu_data = generateRandomInput(latent_dims);
    auto logvar_data = generateRandomInput(latent_dims);
    auto z_data = generateRandomInput(latent_dims);

    // Run inference on the full VAE model
    {
        const char* input_names[] = {"input"};
        const char* output_names[] = {"reconstructed", "mu", "logvar"};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_dims.data(), input_dims.size());

        auto output_tensors = vae_full_session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3);

        std::cout << "Full VAE inference outputs:\n";
        std::cout << "  Reconstructed shape: ";
        for (auto dim : output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << "\n  Mu shape: ";
        for (auto dim : output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << "\n  LogVar shape: ";
        for (auto dim : output_tensors[2].GetTensorTypeAndShapeInfo().GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << "\n";
    }

    // Run inference on the encoder model
    {
        const char* input_names[] = {"input"};
        const char* output_names[] = {"mu", "logvar"};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_dims.data(), input_dims.size());

        auto output_tensors = encoder_session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);

        std::cout << "Encoder inference outputs:\n";
        std::cout << "  Mu shape: ";
        for (auto dim : output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << "\n  LogVar shape: ";
        for (auto dim : output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << "\n";
    }

    // Run inference on the reparameterize model
    {
        const char* input_names[] = {"mu", "logvar"};
        const char* output_names[] = {"z"};

        Ort::Value mu_tensor = Ort::Value::CreateTensor<float>(memory_info, mu_data.data(), mu_data.size(), latent_dims.data(), latent_dims.size());
        Ort::Value logvar_tensor = Ort::Value::CreateTensor<float>(memory_info, logvar_data.data(), logvar_data.size(), latent_dims.data(), latent_dims.size());
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(std::move(mu_tensor));
        input_tensors.emplace_back(std::move(logvar_tensor));

        auto output_tensors = reparameterize_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 2, output_names, 1);

        std::cout << "Reparameterize inference outputs:\n";
        std::cout << "  Z shape: ";
        for (auto dim : output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << "\n";
    }

    // Run inference on the decoder model
    {
        const char* input_names[] = {"z"};
        const char* output_names[] = {"reconstructed"};

        Ort::Value z_tensor = Ort::Value::CreateTensor<float>(memory_info, z_data.data(), z_data.size(), latent_dims.data(), latent_dims.size());

        auto output_tensors = decoder_session.Run(Ort::RunOptions{nullptr}, input_names, &z_tensor, 1, output_names, 1);

        std::cout << "Decoder inference outputs:\n";
        std::cout << "  Reconstructed shape: ";
        for (auto dim : output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nONNX inference completed successfully!\n";
    return 0;
}
