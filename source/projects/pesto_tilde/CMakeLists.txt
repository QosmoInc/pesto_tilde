# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.
cmake_minimum_required(VERSION 3.10)

# Set CMake policy CMP0144 to NEW to allow TORCH_ROOT to be recognized
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()


set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
endif()


#############################################################
# MAX EXTERNAL
#############################################################


include_directories( 
	"${C74_INCLUDES}"
	"${TORCH_INSTALL_PREFIX}/include"
	"${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
)

# Set torch install prefix to the local libtorch directory
set(TORCH_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch")
message(STATUS "Using libtorch from: ${TORCH_INSTALL_PREFIX}")

# Tell CMake where to find Torch
set(Torch_DIR "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")
message(STATUS "Torch_DIR set to: ${Torch_DIR}")

# Find libtorch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set( SOURCE_FILES
	${PROJECT_NAME}.cpp
)


add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)

# Link against torch libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

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

# Copy torch libraries to the output directory
if (APPLE)
  # Create a file list variable to handle paths with spaces
  file(GLOB TORCH_DYLIBS "${TORCH_INSTALL_PREFIX}/lib/*.dylib")
  
  foreach(DYLIB ${TORCH_DYLIBS})
    add_custom_command(
      TARGET ${PROJECT_NAME}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DYLIB}"
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/"
      COMMENT "Copying ${DYLIB}"
    )
  endforeach()
  
  # Fix libomp.dylib identifier
  add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND install_name_tool -id "@loader_path/libomp.dylib"
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/libomp.dylib"
    COMMENT "Fixing libomp.dylib identifier"
  )
  
  # Fix libomp.dylib reference in libtorch_cpu.dylib
  add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND install_name_tool -change "/opt/homebrew/opt/libomp/lib/libomp.dylib" "@loader_path/libomp.dylib"
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/libtorch_cpu.dylib"
    COMMENT "Fixing libomp path in libtorch_cpu.dylib"
  )
  
  # Make sure libomp.dylib is executable
  add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND chmod +x
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/libomp.dylib"
    COMMENT "Making libomp.dylib executable"
  )

  if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    add_custom_command( 
      TARGET ${PROJECT_NAME} 
      POST_BUILD 
      COMMAND "codesign" "--force" "--deep" "-s" "-" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
      COMMENT "Codesign external" 
    )
  endif()
endif()

if (MSVC)
  # Define the support directory path relative to the project root
  set(SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../support")
  message(STATUS "Support directory set to: ${SUPPORT_DIR}")
  
  # Create support directory if it doesn't exist
  add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${SUPPORT_DIR}"
    COMMENT "Creating support directory if it doesn't exist"
  )

  # Copy all torch DLLs to both externals and support directories
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  
  foreach(DLL ${TORCH_DLLS})
    # Copy to support directory
    add_custom_command(
      TARGET ${PROJECT_NAME}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DLL}"
      "${SUPPORT_DIR}/"
      COMMENT "Copying ${DLL} to support"
    )
  endforeach()

endif()
