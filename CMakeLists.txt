cmake_minimum_required(VERSION 3.16)
project(counting_cards)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch") 

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Torch_DIR set to: ${Torch_DIR}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


set(LIB_CV
    src/preprocess.cpp
    src/process.cpp
    src/detect.cpp
)

add_library(cv STATIC ${LIB_CV})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

add_executable(cv_detection src/cv_detection.cpp)

target_include_directories(cv PRIVATE ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_include_directories(cv_detection PRIVATE ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(cv_detection PRIVATE cv ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_property(TARGET cv_detection PROPERTY CXX_STANDARD 17)
