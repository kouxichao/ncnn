#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#add_definitions(-DARM_HISI)

#crnn_chinese
add_executable(demo_chinese crnn_chinese.cpp)
target_link_libraries(demo_chinese ncnn dlib::dlib ${OpenCV_LIBS})


#crnn_english
ADD_LIBRARY(crnn_english STATIC interface_crnn.cpp edit_dis.cpp)
add_executable(demo_english demo_crnn.cpp)
target_link_libraries(demo_english crnn_english ncnn dlib::dlib)
