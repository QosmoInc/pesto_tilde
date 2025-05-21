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
namespace fs = std::filesystem;

using namespace c74::min;


class pesto : public object<pesto>, public vector_operator<> {
public:
    MIN_DESCRIPTION	{"A wrapper to run streaming inference on PESTO pitch estimation models."};
    MIN_TAGS		{"audio, machine learning, inference"};
    MIN_AUTHOR		{"Tom Baker @ Qosmo, Japan"};
    MIN_RELATED		{"fzero~, fiddle~, sigmund~"};

    // Initial chunk size argument that determines target model size at initialization
    argument<number> init_chunk {this, "init_chunk",
        description { "Model chunk size (in samples). Specifying a size will load a matching model from pesto/models. Use 0 to load the fastest available model." },
        MIN_ARGUMENT_FUNCTION {
            m_target_chunk = arg;
            // Initialize model with the specified chunk size
            initialize_model();
        }
    };

    // This message is called once the object is fully constructed
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {         
            cout << "PESTO model - by Alain Riou @ Sony CSL, Paris" << endl;
            cout << "Max external - by Tom Baker @ Qosmo, Japan" << endl;
            return {};
        }
    };

    // Message to change the model at runtime
    message<> model { this, "model", "Load a specific model by filename from pesto/models directory (e.g., 'model 20251502_sr44k_h512.pt'). Can be used to change models at runtime.",
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
    message<> chunk { this, "chunk", "Load a model based on chunk size (e.g., 'chunk 512'). Searches for and loads a model matching the specified chunk size from pesto/models.",
        MIN_FUNCTION {
            m_target_chunk = args[0];
            m_model_path = symbol(""); // Reset model path to force reloading
            initialize_model();
            return {};
        }
    };

    inlet<>  input	{ this, "(signal) audio input, (bang) force model inference" };
    outlet<> pitch_output	{ this, "(float) model's pitch prediction in MIDI note number" };
    outlet<> confidence_output	{ this, "(float) model's confidence prediction (0-1)" };
    outlet<> amplitude_output	{ this, "(float) model's amplitude prediction (0-1)" };

    // Confidence threshold
    attribute<number> conf { this, "conf", 0.0,
        description { "Confidence threshold (0-1). When set, pitch output will be -1500 if confidence is below threshold" },
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
        description { "Amplitude threshold (0+). When set, pitch output will be -1500 if amplitude is below threshold" },
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

    message<> bang { this, "bang", "Force immediate model inference with current audio data, then clear buffers",
        MIN_FUNCTION {
            // Force immediate inference with current data
            m_force_inference = true;
            deliverer.delay(0);

            // Clear the buffer after inference
            clear_buffer();
            feed_zeros_to_model();
            return {};
        }
    };
    
    message<> reset { this, "reset", "Reset the object by clearing both audio and model internal buffers",
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
    
    message<> dspstate { this, "dspstate",
        MIN_FUNCTION {
            long state = args[0];
            
            if (state == 0) { // DSP off
                m_dsp_active = false;
                clear_buffer();
                feed_zeros_to_model();
            }
            else if (state == 1) { // DSP on
                m_dsp_active = true;
            }
            
            return {};
        }
    };

    message<> test { this, "test", "Run model inference on random test data and report the inference latency",
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
                
                // Run the model inference
                auto outputs = m_module.forward(inputs);
                
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

    message<> freq { this, "freq", "Test model with a single chunk of sine wave input at specified frequency (Hz). Usage: 'freq 440'",
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
            auto outputs = m_module.forward(inputs);
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

    timer<> deliverer { this, 
        MIN_FUNCTION {
            // Move to main thread (non-audio thread)
            deferrer.set();
            return {};
        }
    };
    
    queue<> deferrer { this, 
        MIN_FUNCTION {
            // Run inference if we have a full buffer or forced
            if ((m_current_samples >= n_chunk_size) || m_force_inference) {
                run_inference();
                m_force_inference = false;
            }
            return {};
        }
    };

    // Process incoming audio and collect into buffer
    void operator()(audio_bundle input, audio_bundle output) {
        auto in = input.samples(0);
        
        for (auto i = 0; i < input.frame_count(); ++i) {
            // Add sample to the FIFO
            m_audio_fifo.try_enqueue(in[i]);
            m_current_samples++;
            
            // When buffer has enough samples, trigger inference
            // This will happen more frequently with smaller chunk sizes
            if (m_current_samples >= n_chunk_size) {
                deliverer.delay(0);
                // Don't reset m_current_samples here - it's reset in run_inference()
            }
        }
    }

    pesto() {
        // Disable gradient computation for better inference performance
        static torch::NoGradGuard no_grad;
        
        m_model_loaded = false;
        n_chunk_size = 512; 
        m_current_samples = 0;
        m_force_inference = false;
        m_samplerate = 44100.0;
        m_dsp_active = false;
        m_confidence_threshold = 0.0;
        m_amplitude_threshold = 0.0;
        m_model_path = symbol("");
        m_target_chunk = 0; 
        m_saved_phase = 0.0f; 
    }

    // Initialize the model based on current settings
    void initialize_model() {
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
    }

    // Add this method to get the package models directory
    std::string get_models_directory() {
        try {
            // Get the path to the external
            path external_path = path("pesto~", path::filetype::external);
            std::string path_str = external_path;
            
            if (external_path) {
                // Go up two directories from the external to reach the package root
                fs::path package_path = fs::path(path_str).parent_path().parent_path();
                // Add models directory
                fs::path models_path = package_path / "models";
                
                // Create models directory if it doesn't exist
                if (!fs::exists(models_path)) {
                    fs::create_directories(models_path);
                }
                
                return models_path.string();
            }
        } catch (const std::exception& e) {
            cout << "Error getting models directory: " << e.what() << endl;
        }
        return "";
    }

    // Modify the load_model function to only look in the package's models directory
    bool load_model(const std::string& model_file_str) {
        try {
            if (model_file_str.empty()) {
                cout << "Model path is empty" << endl;
                return false;
            }

            std::string models_dir = get_models_directory();
            if (models_dir.empty()) {
                return false;
            }

            // Always look in the models directory
            std::string full_path = models_dir + "/" + model_file_str;
            m_module = torch::jit::load(full_path);
            m_module.eval();
            
            // Try to extract chunk size from filename if possible
            try {
                std::regex pattern(".*h(\\d+)\\.pt$");
                std::smatch matches;
                if (std::regex_search(model_file_str, matches, pattern) && matches.size() > 1) {
                    n_chunk_size = std::stoi(matches[1].str());
                }
            } catch (...) {
                // Keep default chunk size if extraction fails
            }
            
            cout << "Model loaded successfully - Chunk size = " << n_chunk_size << endl;
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
        std::string models_dir = get_models_directory();
        
        if (models_dir.empty()) return compatible_models;

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
    int m_current_samples;      // Current count of samples collected
    fifo<float> m_audio_fifo {16384}; // Thread-safe FIFO for audio samples
    bool m_force_inference;     // Flag for manual inference triggering
    number m_samplerate;        // Current sample rate
    int m_vectorsize;           // Current vector size
    bool m_dsp_active;          // Flag indicating if DSP is active
    number m_confidence_threshold; // Confidence threshold for pitch output
    number m_amplitude_threshold;  // Amplitude threshold for pitch output
    symbol m_model_path;        // Path to model specified by argument
    number m_target_chunk;      // Target chunk size for model initialization
    float m_saved_phase = 0.0f; // Keep track of phase for frequency tests
    
    torch::jit::script::Module m_module;
    bool m_model_loaded;
    
    // Clear the audio buffer and reset the sample counter
    void clear_buffer() {
        m_current_samples = 0;
        while (m_audio_fifo.size_approx() > 0) {
            float temp;
            m_audio_fifo.try_dequeue(temp);
        }
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
                m_module.forward(inputs);
            }
        }
        catch (const c10::Error& e) {
            cout << "Error feeding zeros to model: " << e.what() << endl;
        }
    }
    
    // Run inference on the collected audio samples
    void run_inference() {
        if (!m_model_loaded || m_current_samples < n_chunk_size) {
            cout << "Model not loaded!" << endl;
            return;
        }
        
        try {
            // Transfer audio from FIFO to a contiguous buffer
            std::vector<float> audio_buffer(n_chunk_size);
            
            for (int i = 0; i < n_chunk_size; i++) {
                if (!m_audio_fifo.try_dequeue(audio_buffer[i])) {
                    return;
                }
            }
            
            m_current_samples -= n_chunk_size;
            
            // Create tensor and run model inference
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor input_tensor = torch::from_blob(audio_buffer.data(), {1, (int)n_chunk_size}, options).clone();
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
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
        
        // Check audio FIFO size and adjust processing frequency if needed
        if (m_audio_fifo.size_approx() > n_chunk_size*2) {
            // Buffer is getting ahead, process more frequently
            deliverer.delay(0);         
        }
    }
};


MIN_EXTERNAL(pesto);
