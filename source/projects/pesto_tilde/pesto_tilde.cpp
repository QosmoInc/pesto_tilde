/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <onnxruntime_cxx_api.h>
#include <regex>
#include <vector>
#include <string>
#include <set>
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
private:
    // Constants for better maintainability
    static constexpr float SILENCE_PITCH_VALUE = -1500.0f;
    static constexpr size_t OUTPUT_PREDICTION = 0;
    static constexpr size_t OUTPUT_CONFIDENCE = 1;
    static constexpr size_t OUTPUT_VOLUME = 2;
    static constexpr size_t OUTPUT_ACTIVATIONS = 3;
    static constexpr size_t OUTPUT_CACHE = 4;

    // Inference result structure
    struct InferenceResult {
        float pitch;
        float confidence;
        float amplitude;
        bool success;
    };

public:
    MIN_DESCRIPTION	{"Streaming neural pitch estimation. A Max/MSP wrapper for PESTO, a super Low-latency neural network-based pitch detection model for monophonic audio, providing continuous fundamental frequency estimation as midi values as well as both prediction confidence and note amplitude."};
    MIN_TAGS		{"audio, machine learning, pitch estimation"};
    MIN_AUTHOR		{"Qosmo"};
    MIN_RELATED		{"fzero~, fiddle~, sigmund~"};

    // Model search attributes
    attribute<symbol> checkpoint { this, "checkpoint", symbol("mir-1k_g7"),
        description {"Model checkpoint name (prefix for model files)"}
    };

    attribute<std::vector<int>> sample_rates { this, "sample_rates", {44100, 48000, 96000},
        description {"List of sample rates to search for in models"}
    };

    attribute<std::vector<int>> chunk_sizes { this, "chunk_sizes", {128, 256, 512, 1024},
        description {"List of chunk sizes to search for in models"}
    };

    // Initial chunk size argument that determines target model size at initialization
    // Placed after attributes so they are initialized first
    argument<number> init_chunk {this, "init_chunk", "Specify model chunk size. Specifying a size will load a matching model from pesto/models, this is a required argument", true,
        MIN_ARGUMENT_FUNCTION {
            double chunk_value = static_cast<double>(arg);
            if (chunk_value <= 0) {
                error("Chunk size must be positive");
                return;
            }
            m_target_chunk = chunk_value;
            
            // Debug: check attribute values
            cout << "Checkpoint: " << std::string(checkpoint.get()) << endl;
            cout << "Sample rates size: " << sample_rates.get().size() << endl;
            cout << "Chunk sizes size: " << chunk_sizes.get().size() << endl;
            
            // Attributes should now be initialized to their defaults
            // Find models using the initialized attributes
            find_models();
            
            // Check models found
            if (m_found_models.empty()) {
                error("No models found. Make sure .onnx model files are in Max's search path with naming pattern: " + 
                      std::string(checkpoint.get()) + "_<samplerate>_<chunksize>.onnx");
                return;
            }
            
            // Find best model with requested chunk size
            auto best_match = best_model(m_target_chunk);
            
            if (best_match.filename.empty()) {
                error("No model found with chunk size " + std::to_string(m_target_chunk) + ". " + get_available_chunk_sizes_string());
                return;
            }
            
            // Try to load the model
            if (!load_model(best_match)) {
                error("Failed to load model: " + best_match.filename);
                return;
            }
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
    message<> model { this, "model", "Load a model by filename. Searches for and loads a model matching the specified name (e.g., 'model model_name.onnx') from pesto/models. Can be used to change models at runtime.",
        MIN_FUNCTION {
            if (args.size() > 0) {
                std::string target_path_str = args[0];
                
                // Find the ModelInfo for this filename
                auto it = std::find_if(m_found_models.begin(), m_found_models.end(),
                                    [&target_path_str](const auto& model) { return model.filename == target_path_str; });
                
                if (it != m_found_models.end()) {
                    // Try to load the model
                    if (load_model(*it)) {
                        // Model loaded successfully, update state
                        m_target_chunk = -1; // Clear target chunk
                        m_target_path = args[0]; // Set target path
                    } else {
                        cerr << "Failed to load model: " << target_path_str << endl;
                    }
                } else {
                    cerr << "Model not found: " << target_path_str << ". " << get_available_models_string() << endl;
                }
            }
            return {};
        }
    };
    
    // Message to change the chunk size at runtime
    message<> chunk { this, "chunk", "Load a model by chunk size. Searches for and loads a model matching the specified chunk size (e.g., 'chunk 512') from pesto/models. Can be used to change models at runtime. ",
        MIN_FUNCTION {
            int requested_chunk = args[0];
            auto best_match = best_model(requested_chunk);
            
            if (best_match.filename.empty()) {
                cerr << "No model found with chunk size " << requested_chunk << ". " << get_available_chunk_sizes_string() << endl;
                return {};
            }
            
            // Try to load the model
            if (load_model(best_match)) {
                // Model loaded successfully, update state
                m_target_path = symbol(""); // Clear target path
                m_target_chunk = requested_chunk;
            } else {
                cerr << "Failed to load model for chunk size " << requested_chunk << endl;
            }
            return {};
        }
    };

    inlet<>  input	{ this, "(signal) audio input, (bang) clear buffers and reset model state" };
    
    // Outlets are created dynamically after model loading
    std::unique_ptr<outlet<>> pitch_output;
    std::unique_ptr<outlet<>> confidence_output;
    std::unique_ptr<outlet<>> amplitude_output;

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
            clear_cache();
            return {};
        }
    };
    
    message<> refresh { this, "refresh", "Refresh the models list based on current attribute settings.",
        MIN_FUNCTION {
            find_models();
            cout << "Refreshed models list. Found " << m_found_models.size() << " models." << endl;
            return {};
        }
    };
    
    message<> dspsetup { this, "dspsetup",
        MIN_FUNCTION {
            number new_samplerate = args[0];
            m_vectorsize = args[1];
            m_dsp_active = true;
            
            // Check if sample rate has changed and we have a model loaded
            if (!m_current_model.filename.empty() && new_samplerate != m_samplerate) {
                m_samplerate = new_samplerate;
                
                if (m_target_chunk != -1) {
                    // Loading by chunk size - try to find best model for new sample rate
                    auto best_match = best_model(m_target_chunk);
                    
                    if (!best_match.filename.empty()) {
                        if (!load_model(best_match)) {
                            cout << "Failed to reload model for new sample rate" << endl;
                        }
                    } else {
                        cout << "Error in switching model" << endl;
                    }
                } else {
                    // Loading by model path - can't change to different sample rate
                    cout << "Model loaded has different sample rate" << endl;
                }
            } else {
                m_samplerate = new_samplerate;
            }
            
            // Recalculate sample rate offset if we have a model loaded
            if (!m_current_model.filename.empty()) {
                sr_convert(m_current_model.sample_rate);
            }
            
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

    message<> test { this, "test", "Test inference latency. Run model inference on random test chunk and report the ONNX model's inference latency",
        MIN_FUNCTION {
            if (m_current_model.filename.empty()) {
                cout << "Cannot run test: No model loaded" << endl;
                return {};
            }
            
            // Create random test buffer
            std::vector<float> test_buffer(n_chunk_size);
            for (int i = 0; i < n_chunk_size; i++) {
                test_buffer[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            }
            
            // Measure inference time
            auto start_time = std::chrono::high_resolution_clock::now();
            
            {
                std::lock_guard<std::mutex> lock(m_model_mutex);
                run_model_inference(test_buffer.data(), false); // Don't update cache for test
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            cout << "  Latency: " << duration.count() / 1000.0 << " ms" << endl;
            return {};
        }
    };

    // File usage message for collective/standalone building
    message<> fileusage { this, "fileusage", "Register files for collective/standalone building",
        MIN_FUNCTION {
            if (args.size() > 0) {
                void* w = args[0]; // Handle to the file usage builder
                
                // Register all found model files individually
                for (const auto& model : m_found_models) {
                    c74::max::fileusage_addfile(w, 0, model.filename.c_str(), model.path_id);
                }
            }
            return {};
        }
    };

    // Process incoming audio and collect into buffer
    void operator()(audio_bundle input, audio_bundle output) {
        if (!m_dsp_active) return;
        
        // Return early if no model is loaded
        if (m_current_model.filename.empty()) {
            return;
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

    // Get shared ONNX Runtime environment
    static Ort::Env& get_ort_env() {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pesto_tilde");
        return env;
    }

    pesto() : m_data_ready(0), m_result_ready(1), m_should_stop(false) {
        
        m_target_chunk = -1; // Will be set by argument handler
        m_target_path = symbol(""); // Clear target path for initialization
        n_chunk_size = 512; 
        m_samplerate = c74::max::sys_getsr();
        m_vectorsize = c74::max::sys_getblksize();
        m_dsp_active = false;
        m_confidence_threshold = 0.0;
        m_amplitude_threshold = 0.0;
        m_current_model = {"", 0, 0}; // Empty model info (no model loaded)
        
        // Initialize buffers
        int buffer_size = power_ceil(std::max(n_chunk_size, 4096));
        m_in_buffer.resize(buffer_size);
        m_model_input_buffer = std::make_unique<float[]>(1024); // Max chunk size
        
        // Initialize ONNX Runtime session options
        m_session_options.SetIntraOpNumThreads(1);
        m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Pre-allocate inference objects
        m_audio_shape.resize(2);
        m_cache_shape.resize(2);
        m_input_tensors.reserve(2);
        
        
        // Only start inference thread if model loaded successfully (or no model specified)
        m_inference_thread = std::make_unique<std::thread>([this]() {
            while (!m_should_stop.load()) {
                if (m_data_ready.try_acquire_for(std::chrono::milliseconds(100))) {
                    run_inference();
                    m_result_ready.release();
                }
            }
        });
        
        // Mark object as fully constructed
        m_fully_constructed = true;
    }
    
    ~pesto() {
        // Signal thread to stop
        m_should_stop = true;
        m_data_ready.release(); // Wake up thread
        
        // Wait for thread to finish
        if (m_inference_thread && m_inference_thread->joinable()) {
            m_inference_thread->join();
        }
        
        // Clear session but leave environment alone (let it destruct naturally)
        {
            std::lock_guard<std::mutex> lock(m_model_mutex);
            m_ort_session.reset();
        }
    }


    // Model info: filename, sample_rate, chunk_size
    struct ModelInfo {
        std::string filename;
        int sample_rate;
        int chunk_size;
        short path_id = 0; // Max path ID for file location
    };
    
    // Find models using Max's search path system based on attributes
    void find_models() {
        try {
            m_found_models.clear();
            
            std::string checkpoint_str = std::string(checkpoint.get());
        
        // Iterate through all combinations of sample rates and chunk sizes
        for (int sr : sample_rates.get()) {
            for (int chunk : chunk_sizes.get()) {
                // Construct filename without extension: <checkpoint>_<sr>_<chunk>
                std::string filename_base = checkpoint_str + "_" + std::to_string(sr) + "_" + std::to_string(chunk) + ".onnx";
                
                // Try to locate this file using Max's search path
                char search_filename[c74::max::MAX_PATH_CHARS];
                strcpy(search_filename, filename_base.c_str());
                
                short path_id = 0;
                c74::max::t_fourcc outtype;
                
                cout << "Searching for: " << filename_base << endl;
                
                // Search without file type filter (since 'onnx' is not a known Max type)
                auto result = c74::max::locatefile_extended(search_filename, &path_id, &outtype, nullptr, 0);
                cout << "Search result: " << result << " for " << search_filename << endl;
                
                if (result == 0) {
                    cout << "Found model: " << search_filename << " at path_id: " << path_id << endl;
                    // File found! Store it
                    ModelInfo model_info;
                    model_info.filename = search_filename;
                    model_info.sample_rate = sr;
                    model_info.chunk_size = chunk;
                    model_info.path_id = path_id;
                    
                    m_found_models.push_back(model_info);
                }
            }
        }
        
            // Sort by chunk size for consistent ordering
            std::sort(m_found_models.begin(), m_found_models.end(),
                     [](const auto& a, const auto& b) { return a.chunk_size < b.chunk_size; });
        }
        catch (const std::exception& e) {
            // If find_models fails during construction, just continue with empty model list
            // This can happen if attributes aren't fully initialized yet
            m_found_models.clear();
        }
    }
    
    
    // Find the best model that exactly matches chunk size and has closest sample rate
    ModelInfo best_model(int target_chunk_size) {
        ModelInfo best_match = {"", 0, 0}; // Empty ModelInfo
        int closest_sr_diff = INT_MAX;
        
        for (const auto& model : m_found_models) {
            // Only consider models with exact chunk size match
            if (model.chunk_size == target_chunk_size) {
                int sr_diff = abs(model.sample_rate - static_cast<int>(m_samplerate));
                if (sr_diff < closest_sr_diff) {
                    closest_sr_diff = sr_diff;
                    best_match = model;
                }
            }
        }
        
        return best_match;
    }
    
    // Calculate sample rate conversion offset for MIDI output
    void sr_convert(int model_sr) {
        if (model_sr == static_cast<int>(m_samplerate)) {
            m_sr_offset = 0.0f;
        } else {
            // Formula: 12 * log2(current_sr / model_sr) rounded to the nearest third
            float raw_offset = 12.0f * log2f(static_cast<float>(m_samplerate) / static_cast<float>(model_sr));
            m_sr_offset = std::round(raw_offset * 3.0f) / 3.0f;
            cout << "Compensating for difference in sample rate, performance may degrade slightly" << endl;
        }
    }

    bool load_model(const ModelInfo& model_info) {
        try {
            if (model_info.filename.empty()) {
                cout << "Model filename is empty" << endl;
                return false;
            }
            
            // Check if this model is already loaded
            if (m_current_model.filename == model_info.filename) {
                return true; // Already loaded, no need to reload
            }

            // Find the corresponding FoundModel to get the path ID
            auto found_it = std::find_if(m_found_models.begin(), m_found_models.end(),
                [&model_info](const auto& found) { 
                    return found.filename == model_info.filename; 
                });
            
            if (found_it == m_found_models.end()) {
                cout << "Model not found in search results: " << model_info.filename << endl;
                return false;
            }
            
            // Load model from memory buffer instead of file path
            cout << "Loading model from memory buffer: " << model_info.filename << endl;
            
            // Open file using Max's file system
            c74::max::t_filehandle fh;
            if (c74::max::path_opensysfile(found_it->filename.c_str(), found_it->path_id, &fh, c74::max::READ_PERM) != 0) {
                cout << "Failed to open file: " << model_info.filename << endl;
                return false;
            }
            
            // Get file size
            c74::max::t_ptr_size file_size;
            if (c74::max::sysfile_geteof(fh, &file_size) != 0) {
                cout << "Failed to get file size for: " << model_info.filename << endl;
                c74::max::sysfile_close(fh);
                return false;
            }
            
            cout << "Model file size: " << file_size << " bytes" << endl;
            
            // Allocate buffer and read file
            std::vector<char> buffer(file_size);
            c74::max::t_ptr_size bytes_read = file_size;
            if (c74::max::sysfile_read(fh, &bytes_read, buffer.data()) != 0) {
                cout << "Failed to read file: " << model_info.filename << endl;
                c74::max::sysfile_close(fh);
                return false;
            }
            
            c74::max::sysfile_close(fh);
            cout << "Read " << bytes_read << " bytes from file" << endl;
            
            // Load the new ONNX model from memory buffer with comprehensive error handling
            std::unique_ptr<Ort::Session> new_session;
            try {
                new_session = std::make_unique<Ort::Session>(get_ort_env(), buffer.data(), buffer.size(), m_session_options);
                cout << "ONNX session created successfully from memory buffer" << endl;
            } catch (const Ort::Exception& e) {
                cout << "ONNX Exception: " << e.what() << endl;
                cout << "Error code: " << e.GetOrtErrorCode() << endl;
                return false;
            } catch (const std::bad_alloc& e) {
                cout << "Memory allocation failed: " << e.what() << endl;
                return false;
            } catch (const std::exception& e) {
                cout << "Standard exception: " << e.what() << endl;
                return false;
            } catch (...) {
                cout << "Unknown exception occurred during model loading" << endl;
                return false;
            }
            
            // Use chunk size from ModelInfo
            int new_chunk_size = model_info.chunk_size;
            
            // Get input/output information
            auto input_count = new_session->GetInputCount();
            auto output_count = new_session->GetOutputCount();
            
            // Get cache size from second input shape
            if (input_count >= 2) {
                auto cache_shape = new_session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
                m_cache_size = cache_shape[1]; // Assuming shape is [batch_size, cache_size]
            } else {
                cout << "Error: Model must have at least 2 inputs (audio and cache)" << endl;
                return false;
            }
            
            // Store input/output names properly as strings
            std::vector<std::string> new_input_names;
            std::vector<std::string> new_output_names;
            
            for (size_t i = 0; i < input_count; i++) {
                auto input_name = new_session->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                std::string name_str(input_name.get());
                new_input_names.push_back(name_str);
            }
            
            for (size_t i = 0; i < output_count; i++) {
                auto output_name = new_session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                std::string name_str(output_name.get());
                new_output_names.push_back(name_str);
            }
            
            // Now safely replace the model with proper synchronization
            {
                std::lock_guard<std::mutex> lock(m_model_mutex);
                m_ort_session = std::move(new_session);
                n_chunk_size = new_chunk_size;
                m_input_names = std::move(new_input_names);
                m_output_names = std::move(new_output_names);
                
                // Create pointer arrays for ONNX runtime
                m_input_name_ptrs.clear();
                m_output_name_ptrs.clear();
                for (const auto& name : m_input_names) {
                    m_input_name_ptrs.push_back(name.c_str());
                }
                for (const auto& name : m_output_names) {
                    m_output_name_ptrs.push_back(name.c_str());
                }
                
                // Initialize cache state
                m_cache_state.resize(m_cache_size, 0.0f);
                
                // Update pre-allocated shapes for efficiency
                m_audio_shape[0] = 1;
                m_audio_shape[1] = static_cast<int64_t>(n_chunk_size);
                m_cache_shape[0] = 1;
                m_cache_shape[1] = static_cast<int64_t>(m_cache_size);
            }
            
            cout << "Loaded " << model_info.filename << endl;
            m_current_model = model_info;
            sr_convert(m_current_model.sample_rate); // Calculate sample rate offset
            clear_buffer();
            
            // Create outlets now that model is loaded
            if (!pitch_output) {
                pitch_output = std::make_unique<outlet<>>(this, "(float) model's pitch prediction in MIDI note number");
                confidence_output = std::make_unique<outlet<>>(this, "(float) model's confidence prediction (0-1)");
                amplitude_output = std::make_unique<outlet<>>(this, "(float) model's amplitude prediction");
                cout << "Created model outputs" << endl;
            }
            
            return true;
        }
        catch (const Ort::Exception& e) {
            cout << "Error loading the ONNX model: " << e.what() << endl;
            return false;
        }
    }

    
    // Get a formatted list of available chunk sizes for error messages
    std::string get_available_chunk_sizes_string() {
        std::set<int> unique_chunk_sizes;
        
        // Collect unique chunk sizes
        for (const auto& model : m_found_models) {
            unique_chunk_sizes.insert(model.chunk_size);
        }
        
        std::string result = "Available chunk sizes: ";
        bool first = true;
        for (int chunk_size : unique_chunk_sizes) {
            if (!first) result += ", ";
            result += std::to_string(chunk_size);
            first = false;
        }
        return result;
    }
    
    // Get a formatted list of available model files for error messages
    std::string get_available_models_string() {
        std::string result = "Available models: ";
        for (size_t i = 0; i < m_found_models.size(); ++i) {
            result += m_found_models[i].filename;
            if (i < m_found_models.size() - 1) {
                result += ", ";
            }
        }
        return result;
    }


private:
    int n_chunk_size;           // Size of the audio chunks for model inference
    number m_samplerate;        // Current sample rate
    int m_vectorsize;           // Current vector size
    bool m_dsp_active;          // Flag indicating if DSP is active
    number m_confidence_threshold; // Confidence threshold for pitch output
    number m_amplitude_threshold;  // Amplitude threshold for pitch output
    ModelInfo m_current_model;  // Currently loaded model info (empty filename = no model loaded)
    symbol m_target_path;       // Target model path for model message
    int m_target_chunk;         // Target chunk size for chunk message
    float m_sr_offset = 0.0f;   // Sample rate conversion offset for MIDI output
    
    // Model storage for Max search path system
    std::vector<ModelInfo> m_found_models; // Models found via Max search path
    bool m_fully_constructed = false; // Flag to prevent attribute setters from running during construction
    
    // Threading and buffer components
    CircularBuffer m_in_buffer; // Circular buffer for audio input
    std::unique_ptr<float[]> m_model_input_buffer; // Pre-allocated buffer for model input
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::vector<const char*> m_input_name_ptrs;
    std::vector<const char*> m_output_name_ptrs;
    
    // Pre-allocated inference objects for efficiency
    std::vector<int64_t> m_audio_shape;
    std::vector<int64_t> m_cache_shape;
    std::vector<Ort::Value> m_input_tensors;
    
    // Threading synchronization
    std::binary_semaphore m_data_ready, m_result_ready;
    std::unique_ptr<std::thread> m_inference_thread;
    std::atomic<bool> m_should_stop;
    std::mutex m_model_mutex; // Protect model access
    
    std::unique_ptr<Ort::Session> m_ort_session;
    std::vector<float> m_cache_state;
    size_t m_cache_size;
    Ort::SessionOptions m_session_options;
    Ort::MemoryInfo m_memory_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    
    // Clear the audio buffer
    void clear_buffer() {
        m_in_buffer.clear();
    }
    
    // Common inference helper to eliminate code duplication
    InferenceResult run_model_inference(const float* audio_data, bool update_cache = true) {
        InferenceResult result = {0.0f, 0.0f, 0.0f, false};
        
        if (m_current_model.filename.empty()) {
            return result;
        }
        
        try {
            // Prepare shapes
            m_audio_shape[0] = 1;
            m_audio_shape[1] = static_cast<int64_t>(n_chunk_size);
            m_cache_shape[0] = 1;
            m_cache_shape[1] = static_cast<int64_t>(m_cache_size);
            
            // Create tensors
            auto audio_tensor = Ort::Value::CreateTensor<float>(
                m_memory_info, const_cast<float*>(audio_data), n_chunk_size,
                m_audio_shape.data(), m_audio_shape.size()
            );
            
            auto cache_tensor = Ort::Value::CreateTensor<float>(
                m_memory_info, m_cache_state.data(), m_cache_size,
                m_cache_shape.data(), m_cache_shape.size()
            );
            
            // Prepare input array (reuse vector)
            m_input_tensors.clear();
            m_input_tensors.push_back(std::move(audio_tensor));
            m_input_tensors.push_back(std::move(cache_tensor));
            
            // Run inference
            auto output_tensors = m_ort_session->Run(Ort::RunOptions{nullptr},
                m_input_name_ptrs.data(), m_input_tensors.data(), m_input_tensors.size(),
                m_output_name_ptrs.data(), m_output_name_ptrs.size());
            
            // Extract outputs using constants
            float* prediction_data = output_tensors[OUTPUT_PREDICTION].GetTensorMutableData<float>();
            float* confidence_data = output_tensors[OUTPUT_CONFIDENCE].GetTensorMutableData<float>();
            float* volume_data = output_tensors[OUTPUT_VOLUME].GetTensorMutableData<float>();
            
            result.pitch = prediction_data[0];
            result.confidence = confidence_data[0];
            result.amplitude = volume_data[0];
            result.success = true;
            
            // Update cache state if requested
            if (update_cache) {
                float* cache_out_data = output_tensors[OUTPUT_CACHE].GetTensorMutableData<float>();
                std::copy(cache_out_data, cache_out_data + m_cache_size, m_cache_state.begin());
            }
            
        } catch (const Ort::Exception& e) {
            cout << "Error running ONNX model inference: " << e.what() << endl;
        }
        
        return result;
    }
    
    // Reset the model's cache state (much simpler with ONNX)
    void clear_cache() {
        if (m_current_model.filename.empty()) return;
        
        try {
            std::lock_guard<std::mutex> lock(m_model_mutex);
            // Simply reset the cache state to zeros
            std::fill(m_cache_state.begin(), m_cache_state.end(), 0.0f);
        }
        catch (const std::exception& e) {
            cout << "Error resetting cache state: " << e.what() << endl;
        }
    }
    
    // Run inference on the collected audio samples
    void run_inference() {
        if (m_current_model.filename.empty()) {
            return; // Silently return if no model loaded
        }
        
        // Check if we have enough samples
        if (m_in_buffer.available() < n_chunk_size) {
            return;
        }
        
        // Get samples from circular buffer into pre-allocated buffer
        if (!m_in_buffer.get(m_model_input_buffer.get(), n_chunk_size)) {
            return; // Not enough samples available
        }
        
        // Thread-safe model inference using helper
        std::lock_guard<std::mutex> lock(m_model_mutex);
        auto result = run_model_inference(m_model_input_buffer.get(), true);
        
        if (result.success && pitch_output) {
            // Apply confidence and amplitude thresholds
            if ((m_confidence_threshold > 0.0 && result.confidence < m_confidence_threshold) ||
                (m_amplitude_threshold > 0.0 && result.amplitude < m_amplitude_threshold)) {
                pitch_output->send(SILENCE_PITCH_VALUE);
            } else {
                // Apply sample rate conversion offset to MIDI pitch
                pitch_output->send(result.pitch + m_sr_offset);
            }
            
            confidence_output->send(result.confidence);
            amplitude_output->send(result.amplitude);
        }
    }
};


MIN_EXTERNAL(pesto);
