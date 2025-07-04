cmake_minimum_required(VERSION 3.16)
PROJECT (counting_cards)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED )

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
link_directories( ${CMAKE_BINARY_DIR}/bin )
add_executable( main src/main.cpp )
add_executable( cv_detection src/cv_detection.cpp )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
target_link_libraries(main tinyxml ${OpenCV_LIBS})
target_link_libraries(cv_detection cv ${OpenCV_LIBS})
