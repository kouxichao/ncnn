set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ADD_LIBRARY(BSclassify STATIC BSclassify.cpp)
add_executable(demo demo.cpp)
target_link_libraries(demo BSclassify ncnn dlib::dlib)
