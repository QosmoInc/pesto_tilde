/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <regex>
#include <vector>

using namespace c74::min;


class pesto : public object<pesto>, public vector_operator<> {
public:
    MIN_DESCRIPTION	{"Load and run inference on TorchScript models."};
    MIN_TAGS		{"audio, machine learning, inference"};
    MIN_AUTHOR		{"Your Name"};
    MIN_RELATED		{"nn~"};

    inlet<>  input	{ this, "(signal) audio input, (bang) run model inference" };
    outlet<> pitch_output	{ this, "(float) model's pitch prediction" };
    outlet<> confidence_output	{ this, "(float) model's confidence prediction" };
    outlet<> volume_output	{ this, "(float) model's volume prediction" };

    // Attribute for model path
    attribute<symbol> model_path { this, "model", "",
        description {
            "Path to the TorchScript model file (.pt or .pth)"
        },
        setter { MIN_FUNCTION {
            symbol model_file_path = args[0];
            std::string model_file_str = model_file_path;
            
            // Extract input size from model filename
            std::regex pattern(".*?(\\d+)\\.pt$");
            std::smatch matches;
            if (std::regex_search(model_file_str, matches, pattern) && matches.size() > 1) {
                m_input_dimensions = std::stoi(matches[1].str());
                cout << "Extracted input size from model name: " << m_input_dimensions << endl;
            } else {
                cout << "Could not extract input size from model name, using default: " << m_input_dimensions << endl;
            }
            
            m_model_loaded = load_model(model_file_str);
            return args;
        }}
    };
    
    // respond to the bang message to do something
    message<> bang { this, "bang", "Run model inference and output results.",
        MIN_FUNCTION {
            // Request model inference through the safer threading pattern
            m_force_inference = true;
            deliverer.delay(0);
            return {};
        }
    };

    // post to max window == but only when the class is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "TorchScript model loader/processor loaded" << endl;
            return {};
        }
    };
    
    // Message to reset/clear the buffer
    message<> reset { this, "reset", "Clear the audio buffer",
        MIN_FUNCTION {
            clear_buffer();
            feed_zeros_to_model();
            return {};
        }
    };
    
    // DSP setup - called when audio is initialized
    message<> dspsetup { this, "dspsetup",
        MIN_FUNCTION {
            number samplerate = args[0];
            int vectorsize = args[1];
            
            m_samplerate = samplerate;
            m_vectorsize = vectorsize;
            m_dsp_active = true;
            
            return {};
        }
    };
    
    // Called when DSP is being torn down
    message<> dspstate { this, "dspstate",
        MIN_FUNCTION {
            long state = args[0];
            
            if (state == 0) { // DSP is being turned off
                m_dsp_active = false;
                clear_buffer();
                feed_zeros_to_model();
            }
            else if (state == 1) { // DSP is being turned on
                m_dsp_active = true;
            }
            
            return {};
        }
    };
    
    // Timer to move from audio thread to scheduler thread
    timer<> deliverer { this, 
        MIN_FUNCTION {
            // Move to the low-priority main thread
            deferrer.set();
            return {};
        }
    };
    
    // Queue to run inference in the main thread (low priority)
    queue<> deferrer { this, 
        MIN_FUNCTION {
            // Only run inference if we have enough samples or were forced
            if ((m_current_samples >= m_input_dimensions) || m_force_inference) {
                run_inference();
                m_force_inference = false;
            }
            return {};
        }
    };

    // Process audio vectors
    void operator()(audio_bundle input, audio_bundle output) {
        // Get input samples
        auto in = input.samples(0);
        
        // Process each sample in the vector
        for (auto i = 0; i < input.frame_count(); ++i) {
            // Add the sample to the FIFO
            m_audio_fifo.try_enqueue(in[i]);
            
            // Track number of samples we've collected
            if (m_current_samples < m_input_dimensions) {
                m_current_samples++;
                
                // If we've collected exactly enough samples, schedule an inference
                if (m_current_samples >= m_input_dimensions) {
                    deliverer.delay(0);
                }
            }
        }
        
        // Note: This is a non-audio generator object, so we don't output any audio
    }

    // Constructor
    pesto() {
        m_model_loaded = false;
        m_input_dimensions = 512; // Default input dimensions
        m_current_samples = 0;
        m_force_inference = false;
        m_samplerate = 44100.0; // Default sample rate
        m_dsp_active = false;
        
        cout << "LibTorch version: " << TORCH_VERSION << endl;
    }

private:
    int m_input_dimensions;     // Size of input tensor dimension
    int m_current_samples;      // Current count of samples in the buffer
    fifo<float> m_audio_fifo {16384}; // Thread-safe FIFO for audio samples
    bool m_force_inference;     // Flag for manual inference triggering
    number m_samplerate;        // Current sample rate
    int m_vectorsize;           // Current vector size
    bool m_dsp_active;          // Flag indicating if DSP is active
    
    // TorchScript model components
    torch::jit::script::Module m_module;
    bool m_model_loaded;
    
    // Helper function to clear the audio buffer
    void clear_buffer() {
        m_current_samples = 0;
        while (m_audio_fifo.size_approx() > 0) {
            float temp;
            m_audio_fifo.try_dequeue(temp);
        }
    }
    
    // Helper function to feed zeros to the model to clear its internal state
    void feed_zeros_to_model() {
        if (!m_model_loaded) return;
        
        try {
            // Create a vector of zeros
            std::vector<float> zeros(m_input_dimensions, 0.0f);
            
            // Create tensor options
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            
            // Feed zeros to the model a few times to clear any internal state
            for (int i = 0; i < 3; i++) {  // Feed zeros 3 times
                torch::Tensor zero_tensor = torch::from_blob(zeros.data(), {1, (int)m_input_dimensions}, options).clone();
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(zero_tensor);
                
                m_module.forward(inputs);  // Run inference but don't use the output
            }
        }
        catch (const c10::Error& e) {
            cout << "Error feeding zeros to model: " << e.what() << endl;
        }
    }
    
    // Function to run inference on the current buffer contents
    void run_inference() {
        if (!m_model_loaded || m_current_samples < m_input_dimensions) {
            return;
        }
        
        try {
            // Create a vector to hold the samples from the FIFO
            std::vector<float> audio_buffer(m_input_dimensions);
            
            // Copy samples from the FIFO to the buffer
            for (int i = 0; i < m_input_dimensions; i++) {
                if (!m_audio_fifo.try_dequeue(audio_buffer[i])) {
                    // Not enough samples in the FIFO
                    return;
                }
            }
            
            // Update the current sample count
            m_current_samples -= m_input_dimensions;
            
            // Create a tensor from the audio buffer
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor input_tensor = torch::from_blob(audio_buffer.data(), {1, (int)m_input_dimensions}, options).clone();
            
            // Create inputs for the model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Execute the model
            auto outputs = m_module.forward(inputs);
            auto output_tuple = outputs.toTuple();
            
            // Extract the pitch, confidence, and volume values
            float pitch = output_tuple->elements()[0].toTensor().item<float>();
            float confidence = output_tuple->elements()[1].toTensor().item<float>();
            float volume = output_tuple->elements()[2].toTensor().item<float>();
            
            // Send the values to the outlets (right to left order)
            volume_output.send(volume);
            confidence_output.send(confidence);
            pitch_output.send(pitch);
        }
        catch (const c10::Error& e) {
            cout << "Error running model inference: " << e.what() << endl;
        }
    }
    
    // Function to load a TorchScript model
    bool load_model(const std::string& model_file_str) {
        try {
            // Check if path is empty
            if (model_file_str.empty()) {
                cout << "Model path is empty" << endl;
                return false;
            }
            
            cout << "Loading model from: " << model_file_str << endl;
            
            // Load the TorchScript model
            m_module = torch::jit::load(model_file_str);
            m_module.eval(); // Set to evaluation mode
            
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
