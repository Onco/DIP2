cmake_minimum_required(VERSION 3.5.1)
project( ReFovea )

find_package( OpenCV REQUIRED )

include_directories(.)
set( SRCS ReFovea.cpp )

add_executable( ReFovea ${SRCS} )
target_link_libraries( ReFovea ${OpenCV_LIBS} )
