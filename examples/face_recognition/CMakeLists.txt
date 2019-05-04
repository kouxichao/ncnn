set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ADD_LIBRARY(interface_face STATIC interface_face.cpp)
add_executable(demo_face demo.cpp)
target_link_libraries(demo_face interface_face ncnn dlib::dlib)

if(EVALUATE)
    add_executable(detect_face detect_face.cpp)
    target_link_libraries(detect_face ncnn dlib::dlib)    
    add_executable(evaluate evaluate.cpp)
    target_link_libraries(evaluate interface_face ncnn dlib::dlib)
endif()

