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

using namespace c74::min;


class pesto : public object<pesto>, public vector_operator<> {
public:
    MIN_DESCRIPTION	{"Load and run inference on TorchScript models."};
    MIN_TAGS		{"audio, machine learning, inference"};
    MIN_AUTHOR		{"Your Name"};
    MIN_RELATED		{"nn~"};

    // First argument is required model path
    argument<symbol> model_path { this, "model_path", "Path to TorchScript model (.pt/.pth)",
        MIN_ARGUMENT_FUNCTION {
            std::string model_file_str = arg;
            
            // Extract both sample rate and hop size from filename
            // Format: <date>_sr<samplerate>_h<chunk>.pt
            // e.g., 250515_sr44k_h512.pt
            std::regex pattern(".*sr(\\d+)k.*h(\\d+)\\.pt$");
            std::smatch matches;
            if (std::regex_search(model_file_str, matches, pattern) && matches.size() > 2) {
                // Extract model sample rate (first 2 digits)
                int model_sr_first_digits = std::stoi(matches[1].str());
                
                // Extract chunk size
                n_chunk_size = std::stoi(matches[2].str());
                
                cout << "Extracted from model name - Sample rate: " << model_sr_first_digits << "kHz, Chunk size: " << n_chunk_size << endl;
                
                // Check sample rate matches immediately
                int max_sr_first_digits = static_cast<int>(m_samplerate / 1000);
                
                if (model_sr_first_digits != max_sr_first_digits) {
                    cout << "WARNING: Model sample rate (" << model_sr_first_digits << "kHz) doesn't match Max sample rate (" 
                         << m_samplerate / 1000.0 << "kHz)" << endl;
                } else {
                    cout << "Sample rate verified: Model and Max sample rates match" << endl;
                }
            } else {
                cout << "Could not extract sample rate and chunk size from model name, using default chunk size: " << n_chunk_size << endl;
            }
            
            m_model_loaded = load_model(model_file_str);
        }
    };

    inlet<>  input	{ this, "(signal) audio input, (bang) run model inference" };
    outlet<> pitch_output	{ this, "(float) model's pitch prediction" };
    outlet<> confidence_output	{ this, "(float) model's confidence prediction" };
    outlet<> amplitude_output	{ this, "(float) model's amplitude prediction" };

    // Confidence threshold
    attribute<number> thresh { this, "thresh", 0.0,
        description { "Confidence threshold (0.0-1.0). If set, pitch will output -1500 when confidence is below the threshold" },
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

    message<> bang { this, "bang", "Run model inference and output results.",
        MIN_FUNCTION {
            // Force immediate inference with current data
            m_force_inference = true;
            deliverer.delay(0);
            return {};
        }
    };

    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "TorchScript model loader/processor loaded" << endl;
            return {};
        }
    };
    
    message<> reset { this, "reset", "Clear the audio buffer",
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
            
            if (m_current_samples < n_chunk_size) {
                m_current_samples++;
                
                // When buffer is full, trigger inference
                if (m_current_samples >= n_chunk_size) {
                    deliverer.delay(0);
                }
            }
        }
    }

    pesto() {
        m_model_loaded = false;
        n_chunk_size = 512;  // Default chunk size for the model
        m_current_samples = 0;
        m_force_inference = false;
        m_samplerate = 44100.0;
        m_dsp_active = false;
        m_confidence_threshold = 0.0;
        
        cout << "LibTorch version: " << TORCH_VERSION << endl;
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
            
            // Apply confidence threshold if set
            if (m_confidence_threshold > 0.0 && confidence < m_confidence_threshold) {
                pitch_output.send(-1500.0f);  // Low confidence, output sentinel value
            } else {
                pitch_output.send(pitch);     // Normal or high confidence
            }
            
            confidence_output.send(confidence);
            amplitude_output.send(amplitude);
        }
        catch (const c10::Error& e) {
            cout << "Error running model inference: " << e.what() << endl;
        }
    }
    
    // Load a TorchScript model from file
    bool load_model(const std::string& model_file_str) {
        try {
            if (model_file_str.empty()) {
                cout << "Model path is empty" << endl;
                return false;
            }
            
            cout << "Loading model from: " << model_file_str << endl;
            m_module = torch::jit::load(model_file_str);
            m_module.eval();
            cout << "Model loaded successfully" << endl;
            
            return true;
        }
        catch (const c10::Error& e) {
            cout << "Error loading the model: " << e.what() << endl;
            return false;
        }
    }
};


MIN_EXTERNAL(pesto);
