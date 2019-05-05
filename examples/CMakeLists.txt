
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
#FIND_LIBRARY(DLIB_LIB dlib /usr/local/lib NO_DEFAULT_PATH)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dlib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../sqlite)
#set(NCNN_EXAMPLE_LINK_LIBRARIES ncnn ${OpenCV_LIBS})
#if(NCNN_VULKAN)
#    list(APPEND NCNN_EXAMPLE_LINK_LIBRARIES ${Vulkan_LIBRARY})
#endif()
#
#add_executable(squeezenet squeezenet.cpp)
#target_link_libraries(squeezenet ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(fasterrcnn fasterrcnn.cpp)
#target_link_libraries(fasterrcnn ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(rfcn rfcn.cpp)
#target_link_libraries(rfcn ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(yolov2 yolov2.cpp)
#target_link_libraries(yolov2 ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(yolov3 yolov3.cpp)
#target_link_libraries(yolov3 ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(mobilenetv2ssdlite mobilenetv2ssdlite.cpp)
#target_link_libraries(mobilenetv2ssdlite ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(mobilenetssd mobilenetssd.cpp)
#target_link_libraries(mobilenetssd ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(squeezenetssd squeezenetssd.cpp)
#target_link_libraries(squeezenetssd ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(shufflenetv2 shufflenetv2.cpp)
#target_link_libraries(shufflenetv2 ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(peleenetssd_seg peleenetssd_seg.cpp)
#target_link_libraries(peleenetssd_seg ${NCNN_EXAMPLE_LINK_LIBRARIES})

option(EVALUATE "test demo" ON)
#add_definitions(-DARM_HISI)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../dlib dlib_build)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sqlite sqlite_build)
add_subdirectory(book_classify)
add_subdirectory(crnn)
add_subdirectory(face_recognition)
