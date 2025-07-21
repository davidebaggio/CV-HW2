cmake_minimum_required(VERSION 3.16)
project(counting_cards)
set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
message(STATUS "Torch_DIR set to: ${Torch_DIR}")

find_package(Torch REQUIRED)


find_package(OpenCV REQUIRED )
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")


include_directories(${OpenCV_INCLUDE_DIRS})
link_directories( ${CMAKE_BINARY_DIR}/bin )

set(LIB_TINYXML
	src/tinyxml2.cpp
)

set(HEADER_TINYXML
	include/tinyxml2.hpp
)

set(LIB_CV
	src/preprocess.cpp
	src/process.cpp
	src/detect.cpp
)

set(HEADER_CV
	include/preprocess.hpp
	include/process.hpp
	include/detect.hpp
)

#set(LIB_SRC
#        src/
#)
#
#set(HEADERS
#        include/
#)

add_library(tinyxml STATIC ${LIB_TINYXML} ${HEADER_TINYXML})
add_library(cv STATIC ${LIB_CV} ${HEADER_CV})
#target_link_libraries(image_lib ${OpenCV_LIBS})

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_executable( main src/main.cpp )
add_executable( cv_detection src/cv_detection.cpp )
add_executable( card_pred_test src/card_pred_test.cpp )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
target_link_libraries(main tinyxml ${OpenCV_LIBS})
target_link_libraries(cv_detection cv ${OpenCV_LIBS})
target_link_libraries(card_pred_test  ${TORCH_LIBRARIES} ${OpenCV_LIBS})