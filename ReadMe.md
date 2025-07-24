# 🌿 pesto~ for Max/MSP

This repository hosts `pesto~`, a Max/MSP external object designed for streaming pitch estimation using the PESTO model. 

[PESTO](https://github.com/SonyCSLParis/pesto) (Pitch Estimation with Self-supervised Transposition-equivariant Objective), developed by Alain Riou et al. at Sony CSL Paris, is a deep learning model for fundamental frequency estimation. This Max wrapper enables musicians, developers, and researchers to integrate PESTO's robust, low-latency (typically under 5ms) pitch tracking directly into their Max patches. Whether for interactive music systems, audio analysis tools, jitter visuals, or live performance.

---

## Getting Started

To use `pesto~` with a pre-built release, first download the latest release from the releases page and unzip it into your Max packages folder. For Max 9, this is typically located at:

*   macOS: `~/Documents/Max 9/Packages/`
*   Windows: `C:\Users\[YourUsername]\Documents\Max 9\Packages\`

The release comes pre-bundled with a handful of models, 128, 256, 512, and 1024 samples at 44.1kHz (the default for Max and Ableton) and 48kHz sample rate. The preprocessing step within the PESTO model is sensitive to sample rate, so we have included the common sample rates. If you use a different sample rate to those specified, the external does some internal mathematics to calculate an offset to shift the MIDI pitch to the correct pitch. However If you would like the model to run natively at the different sample rate, or would just like a different chunk size, you can export more traced models from my ONNX export repository. See the Exporting Models section below for more details. 

## Usage

To use `pesto~`, create a new object in Max with the syntax `pesto~ <chunk_size>`. The chunk size determines how many samples are processed at once - smaller chunks reduce latency while larger chunks improve accuracy. We recommend values between 128 samples (minimum stable size) and 1024 samples (acceptable latency for most applications). You must specify a chunk size argument to load an initial model!

`pesto~` has three outlets:
1.   **Pitch:** Outputs the estimated midi pitch value.
2.   **Confidence:** Outputs the confidence of the pitch estimation, ranging from 0 to 1.
3.   **Amplitude:** Outputs an continuous note amplitude.

You can hotswap models during runtime by sending these messages to the object:
- `chunk <chunk_size>` to adjust the processing chunk size
- `model <modelname.onnx>` to load a specific model file

`pesto~` will continuously output pitch, even if the confidence and amplitude are both very low, so we also include a couple of useful attributes: `@conf <value>` and `@amp <value>`. These provide automatic confidence and amplitude thresholding, returning a heavily negative midi value from the pitch outlet when the confidence or amplitude is below the specified value.

All functionallity is available in the reference, and an example help patch is included in the `help` folder.

### Apple Silicon Macs: Unquarantine (If Needed)

If you encounter issues with Max not loading the external on an Apple Silicon Mac due to security restrictions, you may need to unquarantine the `.mxo` file. This can be done within the pop-up, alternatively you can open Terminal and run:

```bash
xattr -r -d com.apple.quarantine externals/pesto~.mxo/Contents/MacOS/*  
```

---

### Exporting Models

To use `pesto~` with different sample rates or chunk sizes not included in the pre-built release, you need to export new ONNX models from the [my fork](https://github.com/TeeJayBaker/pesto/tree/feature/onnx-export) of the original PESTO repository. Follow the instructions below or refer to the readme in the repository for more details.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/TeeJayBaker/pesto.git
    git checkout feature/onnx-export
    ```
2.  **Export the Model:**
    Run the `export_onnx.py` script, specifying the checkpoint, your desired sample rate (in Hz) and chunk size (hop size in samples). For example, to export a model for a 44.1kHz sample rate and a chunk size of 512 samples:
    ```bash
    python -m realtime.export_onnx 'mir-1k_g7' --sr 44100 --hop 512
    ```
    This will create a `.onnx` file in the root directory, named according to the convention `<CHECKPOINT>_<SAMPLE_RATE>_<CHUNK_SIZE>.onnx` (e.g., `mir-1k_g7_44100_512.onnx`).
3.  **Place Models:**
    Move the exported `.onnx` model file(s) into the `misc` folder within this `pesto~` package's directory (e.g., `.../Packages/pesto/misc/`).
4.  **Load the Model in Max:**
    In Max, you can load the new model by sending the message `model <modelname.onnx>` to the `pesto~` object.

---

## Building from Source

Building `pesto~` from source involves cloning the repository, downloading dependencies (CMake), and then running the specific CMake commands to your OS.

### 1. Clone the Repository
If you haven't already, clone this repository to your local machine. It's recommended to clone it directly into your Max Packages folder. For Max 9, the typical locations are:
*   macOS: `~/Documents/Max 9/Packages/`
*   Windows: `C:\Users\[YourUsername]\Documents\Max 9\Packages\`

Open your terminal or command prompt, navigate to your chosen directory (e.g., inside `Max 9/Packages/`), and run the following command to clone the repository and its submodules:

```bash
git clone https://github.com/QosmoInc/pesto_tilde.git --recursive
```

### 2. CMake Build Steps

Open a terminal or command prompt in the packages's root directory.


**For Apple Silicon Macs:**

```bash
mkdir build
cd build
cmake -DCMAKE_OSX_ARCHITECTURES=arm64 ..
make
```

**For Windows (Visual Studio):**

Ensure you have CMake and Visual Studio (with C++ development tools) installed.

```bash
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```
(Adjust the Visual Studio version in the generator flag `-G` if you have a different version installed.)

This will generate the `pesto~.mxo` (macOS) or `pesto~.mxe64` (Windows) file in the externals folder. If you cloned the repository into your Max 9 packages folder, the external should be immediately available in Max. Otherwise, copy the `externals`, `misc`, `docs` and `help` folders to a new folder in your Max 9 packages folder.

---

## Contributions

Contributions, bug reports, and optimisations are welcome! Especially builds for Pure Data! Please feel free to open an issue or submit a pull request.

## Credits

*   **PESTO Model:**
    Alain Riou, Stefan Lattner, Gaëtan Hadjeres, and Geoffroy Peeters. "PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective." *ISMIR - International Society for Music Information Retrieval*. 2023.
    [https://github.com/SonyCSLParis/pesto](https://github.com/SonyCSLParis/pesto)

