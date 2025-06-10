/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <regex>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <semaphore>
#include <atomic>
#include <chrono>
namespace fs = std::filesystem;

using namespace c74::min;

// Power-of-2 ceiling function for efficient circular buffer sizing
unsigned power_ceil(unsigned x) {
    if (x <= 1) return 1;
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}

// Circular buffer class for efficient audio buffering
class CircularBuffer {
private:
    std::vector<float> buffer;
    std::atomic<size_t> write_pos{0};
    std::atomic<size_t> read_pos{0};
    size_t capacity;
    size_t mask; // For power-of-2 optimization
    
public:
    CircularBuffer() : capacity(0), mask(0) {}
    
    void resize(size_t size) {
        capacity = power_ceil(size);
        mask = capacity - 1;
        buffer.resize(capacity);
        write_pos = 0;
        read_pos = 0;
    }
    
    void put(float sample) {
        size_t pos = write_pos.load(std::memory_order_relaxed) & mask;
        buffer[pos] = sample;
        write_pos.store(write_pos.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }
    
    bool get(float* dest, size_t count) {
        if (available() < count) return false;
        
        size_t pos = read_pos.load(std::memory_order_relaxed) & mask;
        for (size_t i = 0; i < count; ++i) {
            dest[i] = buffer[(pos + i) & mask];
        }
        read_pos.store(read_pos.load(std::memory_order_relaxed) + count, std::memory_order_release);
        return true;
    }
    
    size_t available() const {
        return write_pos.load(std::memory_order_acquire) - read_pos.load(std::memory_order_acquire);
    }
    
    void clear() {
        write_pos = 0;
        read_pos = 0;
    }
};


class pesto : public object<pesto>, public vector_operator<> {
public:
    MIN_DESCRIPTION	{"Streaming neural pitch estimation. A Max/MSP wrapper for PESTO, a super Low-latency neural network-based pitch detection model for monophonic audio, providing continuous fundamental frequency estimation as midi values as well as both prediction confidence and note amplitude."};
    MIN_TAGS		{"audio, machine learning, pitch estimation"};
    MIN_AUTHOR		{"Qosmo"};
    MIN_RELATED		{"fzero~, fiddle~, sigmund~"};

    // Initial chunk size argument that determines target model size at initialization
    argument<number> init_chunk {this, "init_chunk", "Specify model chunk size. Specifying a size will load a matching model from pesto/models. Use 0 to load the fastest available model.", true,
        MIN_ARGUMENT_FUNCTION {
            m_target_chunk = arg;
            // Initialize model with the specified chunk size
            initialize_model();
        }
    };

    // This message is called once the object is fully constructed
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {         
            cout << "PESTO model - by Alain Riou @ Sony CSL Paris" << endl;
            cout << "Max external - by Tom Baker @ Qosmo" << endl;
            return {};
        }
    };

    // Message to change the model at runtime
    message<> model { this, "model", "Load a model by filename. Searches for and loads a model matching the specified name (e.g., 'model 20251502_sr44k_h512.pt') from pesto/models. Can be used to change models at runtime.",
        MIN_FUNCTION {
            if (args.size() > 0) {
                m_model_path = args[0];
                cout << "Model path changed to: " << m_model_path << endl;
                initialize_model();
            }
            return {};
        }
    };
    
    // Message to change the chunk size at runtime
    message<> chunk { this, "chunk", "Load a model by chunk size. Searches for and loads a model matching the specified chunk size (e.g., 'chunk 512') from pesto/models. Can be used to change models at runtime. ",
        MIN_FUNCTION {
            m_target_chunk = args[0];
            m_model_path = symbol(""); // Reset model path to force reloading
            initialize_model();
            return {};
        }
    };

    inlet<>  input	{ this, "(signal) audio input, (bang) clear buffers and reset model state" };
    outlet<> pitch_output	{ this, "(float) model's pitch prediction in MIDI note number" };
    outlet<> confidence_output	{ this, "(float) model's confidence prediction (0-1)" };
    outlet<> amplitude_output	{ this, "(float) model's amplitude prediction" };

    // Confidence threshold
    attribute<number> conf { this, "conf", 0.0,
        description { "Confidence threshold (0-1). If not set the model will continuously output pitch, when set, pitch output will be -1500 if confidence is below threshold" },
        setter { MIN_FUNCTION {
            number threshold = args[0];
            
            // Validate the threshold is in valid range
            if (threshold < 0.0)
                m_confidence_threshold = 0.0;
            else if (threshold > 1.0)
                m_confidence_threshold = 1.0;
            else
                m_confidence_threshold = threshold;
            
            return {};
        }}
    };

    attribute<number> amp { this, "amp", 0.0,
        description { "Amplitude threshold (0+). If not set the model will continuously output pitch, when set, pitch output will be -1500 if amplitude is below threshold" },
        setter { MIN_FUNCTION {
            number threshold = args[0];
            
            // Validate the threshold is in valid range
            if (threshold < 0.0)
                m_amplitude_threshold = 0.0;
            else
                m_amplitude_threshold = threshold;
            
            return {};
        }}
    };

    message<> bang { this, "bang", "Reset the object by clearing buffers. Reset the object by clearing both the Max external's and the PESTO model's internal circular buffer.",
        MIN_FUNCTION {
            clear_buffer();
            feed_zeros_to_model();
            return {};
        }
    };
    
    message<> dspsetup { this, "dspsetup",
        MIN_FUNCTION {
            m_samplerate = args[0];
            m_vectorsize = args[1];
            m_dsp_active = true;
            
            return {};
        }
    };
    
    message<> dspstate { this, "dspstate", "Set the DSP state to either on (1) or off (0).",
        MIN_FUNCTION {
            long state = args[0];
            
            if (state == 0) { // DSP off
                m_dsp_active = false;
                clear_buffer();
            }
            else if (state == 1) { // DSP on
                m_dsp_active = true;
            }
            
            return {};
        }
    };

    message<> test { this, "test", "Test inference latency. Run model inference on random test chunk and report the TorchScript model's inference latency",
        MIN_FUNCTION {
            if (!m_model_loaded) {
                cout << "Cannot run test: No model loaded" << endl;
                return {};
            }
            
            try {
                // Create a vector of random samples for testing
                std::vector<float> test_buffer(n_chunk_size);
                for (int i = 0; i < n_chunk_size; i++) {
                    test_buffer[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Random values between -1.0 and 1.0
                }
                
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                torch::Tensor input_tensor = torch::from_blob(test_buffer.data(), {1, (int)n_chunk_size}, options).clone();
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Measure inference time
                auto start_time = std::chrono::high_resolution_clock::now();
                
                // Run the model inference with mutex protection
                torch::jit::IValue outputs;
                {
                    std::lock_guard<std::mutex> lock(m_model_mutex);
                    outputs = m_module.forward(inputs);
                }
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                
                // Extract model outputs
                auto output_tuple = outputs.toTuple();
                float pitch = output_tuple->elements()[0].toTensor().item<float>();
                float confidence = output_tuple->elements()[1].toTensor().item<float>();
                float amplitude = output_tuple->elements()[2].toTensor().item<float>();
                
                // Print results
                cout << "  Latency: " << duration.count() / 1000.0 << " ms" << endl;
            }
            catch (const c10::Error& e) {
                cout << "Error during test inference: " << e.what() << endl;
            }
            
            return {};
        }
    };

    message<> freq { this, "freq", "Test with a chunk of sinusoidal audio. Test model with a single chunk of sine wave input at specified frequency (Hz) to test accuracy. Usage: 'freq 440'",
    MIN_FUNCTION {
        if (!m_model_loaded) {
            cout << "Cannot run frequency test: No model loaded" << endl;
            return {};
        }
        
        if (args.size() < 1) {
            cout << "Usage: freq [frequency_in_hz]" << endl;
            return {};
        }
        
        float frequency = args[0];
        
        try {
            // Generate sine wave at specified frequency
            std::vector<float> sine_buffer(n_chunk_size);
            float phase = m_saved_phase; // Start from previous phase
            float phase_increment = 2.0f * M_PI * frequency / m_samplerate;
            
            for (int i = 0; i < n_chunk_size; i++) {
                sine_buffer[i] = sin(phase);
                phase += phase_increment;
                if (phase > 2.0f * M_PI) {
                    phase -= 2.0f * M_PI;
                }
            }

            // Save the phase for next call
            m_saved_phase = phase;
            
            // Create tensor and run model inference
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor input_tensor = torch::from_blob(sine_buffer.data(), {1, (int)n_chunk_size}, options).clone();
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Run model inference with mutex protection
            torch::jit::IValue outputs;
            {
                std::lock_guard<std::mutex> lock(m_model_mutex);
                outputs = m_module.forward(inputs);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            // Extract model outputs
            auto output_tuple = outputs.toTuple();
            float pitch = output_tuple->elements()[0].toTensor().item<float>();
            float confidence = output_tuple->elements()[1].toTensor().item<float>();
            float amplitude = output_tuple->elements()[2].toTensor().item<float>();
            
            // Output a single line with the results
            // Convert MIDI note to Hz using the formula: Hz = 440 * 2^((midi-69)/12)
            float pitchHz = 440.0f * pow(2.0f, (pitch - 69.0f) / 12.0f);
            cout << "Freq test: input=" << frequency << "Hz, output=" << pitchHz << "Hz, latency=" << duration.count()/1000.0 << "ms" << endl;
        }
        catch (const c10::Error& e) {
            cout << "Error during frequency test: " << e.what() << endl;
        }
        
        return {};
    }
};

    // Process incoming audio and collect into buffer
    void operator()(audio_bundle input, audio_bundle output) {
        if (!m_dsp_active) return;
        
        // Count audio frames regardless of model state
        if (!m_model_loaded) {
            m_audio_frames_without_model += input.frame_count();
            // Only show error after processing audio for ~0.5 second (22050 frames at 44.1kHz)
            if (m_audio_frames_without_model > 22050 && !m_error_reported) {
                cerr << "An instance of 'pesto~' does not have a model loaded." << endl;
                cerr << "Specify a chunk_size with 'pesto~ <chunk_size>'" << endl;
                cerr << "or use 'pesto~ 0' for the smallest available size." << endl;
                m_error_reported = true;
            }
            return; // Don't process audio if no model
        } else {
            // Reset counter when model is loaded
            m_audio_frames_without_model = 0;
            m_error_reported = false;
        }
        
        auto in = input.samples(0);
        
        for (auto i = 0; i < input.frame_count(); ++i) {
            // Add sample to circular buffer
            m_in_buffer.put(in[i]);
        }
        
        // Check if we have enough samples and inference thread is ready
        if (m_in_buffer.available() >= n_chunk_size && m_result_ready.try_acquire()) {
            // Signal inference thread that data is ready
            m_data_ready.release();
        }
    }

    pesto() : m_data_ready(0), m_result_ready(1), m_should_stop(false) {
        // Disable gradient computation for better inference performance
        static torch::NoGradGuard no_grad;
        
        m_model_loaded = false;
        n_chunk_size = 512; 
        m_samplerate = 44100.0;
        m_dsp_active = false;
        m_confidence_threshold = 0.0;
        m_amplitude_threshold = 0.0;
        m_model_path = symbol("");
        m_target_chunk = 0; 
        m_saved_phase = 0.0f; 
        m_error_reported = false;
        m_audio_frames_without_model = 0;
        
        // Initialize buffers
        int buffer_size = power_ceil(std::max(n_chunk_size, 4096));
        m_in_buffer.resize(buffer_size);
        m_model_input_buffer = std::make_unique<float[]>(1024); // Max chunk size
        
        // Pre-create tensor options
        m_tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
        
        // Start inference thread
        m_inference_thread = std::make_unique<std::thread>([this]() {
            while (!m_should_stop.load()) {
                if (m_data_ready.try_acquire_for(std::chrono::milliseconds(100))) {
                    run_inference();
                    m_result_ready.release();
                }
            }
        });
    }
    
    ~pesto() {
        // Signal thread to stop
        m_should_stop = true;
        m_data_ready.release(); // Wake up thread
        
        // Wait for thread to finish
        if (m_inference_thread && m_inference_thread->joinable()) {
            m_inference_thread->join();
        }
    }

    // Initialize the model based on current settings
    void initialize_model() {
        // Temporarily disable model processing during model change
        bool was_loaded = m_model_loaded;
        m_model_loaded = false;
        
        // Clear buffers to prevent size mismatches
        clear_buffer();
        
        // If a specific model path was provided, load it
        if (m_model_path != symbol("")) {
            cout << "Loading specified model: " << m_model_path << endl;
            std::string model_file_str = m_model_path;
            m_model_loaded = load_model(model_file_str);
            
            // If loading fails, fall back to best matching model
            if (!m_model_loaded) {
                cout << "Failed to load specified model, falling back to best match" << endl;
                load_best_model();
            }
        } 
        // Otherwise, find the best matching model based on chunk size preference
        else {
            load_best_model();
        }
        
        // Resize buffer if chunk size changed and model loaded successfully
        if (m_model_loaded) {
            int buffer_size = power_ceil(std::max(n_chunk_size, 4096));
            m_in_buffer.resize(buffer_size);
        }
    }

    // Add this method to get the package models directories
    std::vector<std::string> get_models_directories() {
        std::vector<std::string> directories;
        try {
            // Get the path to the external
            path external_path = path("pesto~", path::filetype::external);
            std::string path_str = external_path;
            
            if (external_path) {
                // Add the directory where the .mxo file is located
                fs::path external_dir = fs::path(path_str).parent_path();
                if (fs::exists(external_dir)) {
                    directories.push_back(external_dir.string());
                }
                
                // Go up two directories from the external to reach the package root
                fs::path package_path = fs::path(path_str).parent_path().parent_path();
                
                // Add both models and other directories
                fs::path models_path = package_path / "models";
                fs::path other_path = package_path / "other";
                
                // Create models directory if it doesn't exist
                if (!fs::exists(models_path)) {
                    fs::create_directories(models_path);
                }
                
                // Add existing directories to the list
                if (fs::exists(models_path)) {
                    directories.push_back(models_path.string());
                }
                if (fs::exists(other_path)) {
                    directories.push_back(other_path.string());
                }
            }
        } catch (const std::exception& e) {
            cout << "Error getting models directories: " << e.what() << endl;
        }
        return directories;
    }

    bool load_model(const std::string& model_file_str) {
        try {
            if (model_file_str.empty()) {
                cout << "Model path is empty" << endl;
                return false;
            }

            auto models_dirs = get_models_directories();
            if (models_dirs.empty()) {
                cout << "No models directories found" << endl;
                return false;
            }

            // Try to find the model in any of the directories
            std::string full_path;
            bool found = false;
            
            for (const auto& dir : models_dirs) {
                std::string candidate_path = dir + "/" + model_file_str;
                if (fs::exists(candidate_path)) {
                    full_path = candidate_path;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                cout << "Model file not found in any models directory: " << model_file_str << endl;
                return false;
            }
            
            // Load the new model first (outside of the critical section)
            torch::jit::script::Module new_module = torch::jit::load(full_path);
            new_module.eval();
            
            // Extract chunk size from filename before updating anything
            int new_chunk_size = n_chunk_size; // Default to current
            try {
                std::regex pattern(".*h(\\d+)\\.pt$");
                std::smatch matches;
                if (std::regex_search(model_file_str, matches, pattern) && matches.size() > 1) {
                    new_chunk_size = std::stoi(matches[1].str());
                }
            } catch (...) {
                // Keep default chunk size if extraction fails
            }
            
            // Now safely replace the model with proper synchronization
            {
                std::lock_guard<std::mutex> lock(m_model_mutex);
                m_module = std::move(new_module);
                n_chunk_size = new_chunk_size;
            }
            
            cout << "Model loaded successfully - Chunk size = " << n_chunk_size << endl;
            clear_buffer();
            return true;
        }
        catch (const c10::Error& e) {
            cout << "Error loading the model: " << e.what() << endl;
            return false;
        }
    }

    // Find all compatible models in the models directory
    std::vector<std::pair<std::string, int>> find_compatible_models() {
        std::vector<std::pair<std::string, int>> compatible_models;
        auto models_dirs = get_models_directories();
        
        if (models_dirs.empty()) return compatible_models;

        for (const auto& models_dir : models_dirs) {
            for (const auto& entry : fs::directory_iterator(models_dir)) {
                if (entry.path().extension() == ".pt") {
                    std::string filename = entry.path().filename().string();
                    std::regex pattern(".*sr(\\d+)k.*h(\\d+)\\.pt$");
                    std::smatch matches;
                    
                    if (std::regex_search(filename, matches, pattern) && matches.size() > 2) {
                        int model_sr = std::stoi(matches[1].str());
                        int chunk_size = std::stoi(matches[2].str());
                        
                        // Check if sample rate matches
                        if (model_sr == static_cast<int>(m_samplerate / 1000)) {
                            compatible_models.push_back({filename, chunk_size});
                        }
                    }
                }
            }
        }
        
        // Sort by chunk size
        std::sort(compatible_models.begin(), compatible_models.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        return compatible_models;
    }

    // Load the best matching model
    void load_best_model() {
        auto compatible_models = find_compatible_models();
        
        if (compatible_models.empty()) {
            cout << "No compatible models found for sample rate " << m_samplerate / 1000 << "kHz" << endl;
            return;
        }

        // Try to find model matching requested chunk size if specified (non-zero)
        int target_chunk = m_target_chunk;
        
        if (target_chunk > 0) {
            auto it = std::find_if(compatible_models.begin(), compatible_models.end(),
                                [target_chunk](const auto& model) { return model.second == target_chunk; });
            
            if (it != compatible_models.end()) {
                m_model_loaded = load_model(it->first);
                return;
            }
            
            cout << "No model found with chunk size " << target_chunk << endl;
        }
        
        // If we reach here, we're falling back to the smallest chunk size
        m_model_loaded = load_model(compatible_models[0].first);
    }

private:
    int n_chunk_size;           // Size of the audio chunks for model inference
    number m_samplerate;        // Current sample rate
    int m_vectorsize;           // Current vector size
    bool m_dsp_active;          // Flag indicating if DSP is active
    number m_confidence_threshold; // Confidence threshold for pitch output
    number m_amplitude_threshold;  // Amplitude threshold for pitch output
    symbol m_model_path;        // Path to model specified by argument
    number m_target_chunk;      // Target chunk size for model initialization
    float m_saved_phase = 0.0f; // Keep track of phase for frequency tests
    bool m_error_reported = false; // Flag for error reporting
    int m_audio_frames_without_model = 0; // Counter for audio frames processed without model
    
    // Threading and buffer components
    CircularBuffer m_in_buffer; // Circular buffer for audio input
    std::unique_ptr<float[]> m_model_input_buffer; // Pre-allocated buffer for model input
    torch::TensorOptions m_tensor_options; // Pre-created tensor options
    
    // Threading synchronization
    std::binary_semaphore m_data_ready, m_result_ready;
    std::unique_ptr<std::thread> m_inference_thread;
    std::atomic<bool> m_should_stop;
    std::mutex m_model_mutex; // Protect model access
    
    torch::jit::script::Module m_module;
    bool m_model_loaded;
    
    // Clear the audio buffer
    void clear_buffer() {
        m_in_buffer.clear();
    }
    
    // Feed zeros to reset the model's internal state
    void feed_zeros_to_model() {
        if (!m_model_loaded) return;
        
        try {
            std::vector<float> zeros(n_chunk_size, 0.0f);
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            
            // Feed zeros multiple times to clear any internal state
            for (int i = 0; i < 8; i++) {
                torch::Tensor zero_tensor = torch::from_blob(zeros.data(), {1, (int)n_chunk_size}, options).clone();
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(zero_tensor);
                
                {
                    std::lock_guard<std::mutex> lock(m_model_mutex);
                    m_module.forward(inputs);
                }
            }
        }
        catch (const c10::Error& e) {
            cout << "Error feeding zeros to model: " << e.what() << endl;
        }
    }
    
    // Run inference on the collected audio samples
    void run_inference() {
        if (!m_model_loaded) {
            return; // Silently return if no model loaded
        }
        
        // Check if we have enough samples
        if (m_in_buffer.available() < n_chunk_size) {
            return;
        }
        
        m_error_reported = false;

        try {
            // Get samples from circular buffer into pre-allocated buffer
            if (!m_in_buffer.get(m_model_input_buffer.get(), n_chunk_size)) {
                return; // Not enough samples available
            }
            
            // Create tensor using pre-allocated buffer and pre-created options
            torch::Tensor input_tensor = torch::from_blob(
                m_model_input_buffer.get(), 
                {1, (int)n_chunk_size}, 
                m_tensor_options
            ).clone();
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Thread-safe model inference
            std::lock_guard<std::mutex> lock(m_model_mutex);
            auto outputs = m_module.forward(inputs);
            auto output_tuple = outputs.toTuple();
            
            // Extract model outputs
            float pitch = output_tuple->elements()[0].toTensor().item<float>();
            float confidence = output_tuple->elements()[1].toTensor().item<float>();
            float amplitude = output_tuple->elements()[2].toTensor().item<float>();
            
            // Apply confidence and amplitude thresholds
            if ((m_confidence_threshold > 0.0 && confidence < m_confidence_threshold) ||
                (m_amplitude_threshold > 0.0 && amplitude < m_amplitude_threshold)) {
                pitch_output.send(-1500.0f);  // Low confidence/amplitude, output sentinel value
            } else {
                pitch_output.send(pitch);     // Normal or high confidence/amplitude
            }
            
            confidence_output.send(confidence);
            amplitude_output.send(amplitude);
        }
        catch (const c10::Error& e) {
            cout << "Error running model inference: " << e.what() << endl;
        }
    }
};


MIN_EXTERNAL(pesto);
