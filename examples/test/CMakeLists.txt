ADD_LIBRARY(cJson STATIC cJSON.c)
add_executable(demo_test test.cpp)
target_link_libraries(demo_test ncnn cJson)# ${OpenCV_LIBS})
