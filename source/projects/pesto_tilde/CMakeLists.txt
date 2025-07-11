# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.
cmake_minimum_required(VERSION 3.10)

# ONNX Runtime integration policies
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)  # For FetchContent timestamps
endif()


set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "13.4")
endif()


#############################################################
# ONNX RUNTIME SETUP
#############################################################

# Include FetchContent module for downloading ONNX Runtime
include(FetchContent)

# Set ONNX Runtime install prefix
set(ONNXRUNTIME_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../../../onnxruntime")

# Determine platform-specific ONNX Runtime download URLs and filenames
if(APPLE)
    set(ONNXRUNTIME_URL "https://github.com/supertone-inc/onnxruntime-build/releases/download/v1.21.1/onnxruntime-osx-universal2-static_lib-1.21.1.tgz")
    set(ONNXRUNTIME_FILENAME "onnxruntime-osx-universal2-static_lib-1.21.1.tgz")
elseif(WIN32)
    set(ONNXRUNTIME_URL "https://github.com/supertone-inc/onnxruntime-build/releases/download/v1.21.1/onnxruntime-win-x64-static_lib-1.21.1.zip")
    set(ONNXRUNTIME_FILENAME "onnxruntime-win-x64-static_lib-1.21.1.zip")
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime")
endif()

# Download and extract ONNX Runtime if not already present
if(NOT EXISTS "${ONNXRUNTIME_INSTALL_PREFIX}")
    message(STATUS "Downloading ONNX Runtime from: ${ONNXRUNTIME_URL}")
    FetchContent_Declare(
        onnxruntime
        URL ${ONNXRUNTIME_URL}
        SOURCE_DIR ${ONNXRUNTIME_INSTALL_PREFIX}
    )
    FetchContent_MakeAvailable(onnxruntime)
else()
    message(STATUS "Using existing ONNX Runtime from: ${ONNXRUNTIME_INSTALL_PREFIX}")
endif()

# Set ONNX Runtime include directories
include_directories(
    "${C74_INCLUDES}"
    "${ONNXRUNTIME_INSTALL_PREFIX}/include"
)

message(STATUS "ONNX Runtime install prefix: ${ONNXRUNTIME_INSTALL_PREFIX}")

set( SOURCE_FILES
	${PROJECT_NAME}.cpp
)


add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)

# Link against ONNX Runtime static libraries
if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE 
        "${ONNXRUNTIME_INSTALL_PREFIX}/lib/libonnxruntime.a"
        "-framework Foundation"
        "-framework CoreML"
    )
elseif(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE 
        "${ONNXRUNTIME_INSTALL_PREFIX}/lib/onnxruntime.lib"
    )
endif()

# Set C++20 standard correctly for all compilers
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)

if(MSVC)
  # Use the correct flag for MSVC to enable C++20
  target_compile_options(${PROJECT_NAME} PRIVATE "/std:c++latest")
else()
  # For GCC/Clang, explicitly set C++20
  target_compile_options(${PROJECT_NAME} PRIVATE "-std=c++20")
endif()

# Set rpath for Mac to look in loader path only, not in absolute paths
if (APPLE)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/"
    LINK_FLAGS "-Wl,-rpath,@loader_path/"
  )
  
  # Clear CMAKE_INSTALL_RPATH to avoid absolute paths
  set(CMAKE_INSTALL_RPATH "")
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

# ONNX Runtime uses static libraries, no dynamic library copying needed for macOS
if (APPLE)

  if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    add_custom_command( 
      TARGET ${PROJECT_NAME} 
      POST_BUILD 
      COMMAND "codesign" "--force" "--deep" "-s" "-" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
      COMMENT "Codesign external" 
    )
  endif()
endif()

# ONNX Runtime uses static libraries, no DLL copying needed for Windows
if (MSVC)
  # No additional setup needed for static libraries
  message(STATUS "Using ONNX Runtime static libraries for Windows")
endif()
