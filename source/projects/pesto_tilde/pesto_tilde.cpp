/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <regex>

using namespace c74::min;


class pesto : public object<pesto>, public sample_operator<1, 0> {
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
            // If a model is loaded, run inference with random data
            if (m_model_loaded) {
                try {
                    // Create example input tensor using the input dimensions
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(torch::ones({1, m_input_dimensions}));
                    
                    cout << "Running inference with input tensor shape: [1, " << m_input_dimensions << "]" << endl;
                    
                    // Execute the model
                    auto outputs = m_module.forward(inputs);
                    auto output_tuple = outputs.toTuple();
                    
                    // Extract the pitch, confidence, and volume values
                    float pitch = output_tuple->elements()[0].toTensor().item<float>();
                    float confidence = output_tuple->elements()[1].toTensor().item<float>();
                    float volume = output_tuple->elements()[2].toTensor().item<float>();
                    // torch::Tensor activations = output_tuple->elements()[3].toTensor();

                    // Output the values to the console
                    cout << "Model outputs - Pitch: " << pitch << ", Confidence: " << confidence << ", Volume: " << volume << endl;
                    // cout << "Activations shape: [" << activations.sizes()[0] << ", " << activations.sizes()[1] << "]" << endl;

                    // Send the values to the outlets (right to left order)
                    volume_output.send(volume);
                    confidence_output.send(confidence);
                    pitch_output.send(pitch);
                }
                catch (const c10::Error& e) {
                    cout << "Error running model inference: " << e.what() << endl;
                }
            }
            
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

    // Dummy audio operator
    void operator()(sample input) {
        // Do nothing for now
    }

    // Constructor
    pesto() {
        m_model_loaded = false;
        m_input_dimensions = 512; // Initialize input dimensions
        
        cout << "LibTorch version: " << TORCH_VERSION << endl;
    }

private:
    int m_input_dimensions;  // Size of input tensor dimension
    
    // TorchScript model components
    torch::jit::script::Module m_module;
    bool m_model_loaded;
    
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
