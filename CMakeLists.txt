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

#set(LIB_SRC
#        src/
#)
#
#set(HEADERS
#        include/
#)

add_library(tinyxml STATIC ${LIB_TINYXML} ${HEADER_TINYXML})
#target_link_libraries(image_lib ${OpenCV_LIBS})

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin )
add_executable( main src/main.cpp )
add_executable( main3 src/main3.cpp )
add_executable( main2 src/main2.cpp src/card_detection.cpp )
add_executable( inference src/inference.cpp )
add_executable( cv_detection src/cv_detection.cpp )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
target_link_libraries(main tinyxml ${OpenCV_LIBS})
target_link_libraries(inference ${OpenCV_LIBS})
target_link_libraries(cv_detection ${OpenCV_LIBS})
target_link_libraries(main2 ${OpenCV_LIBS})
target_link_libraries(main3 ${OpenCV_LIBS})
