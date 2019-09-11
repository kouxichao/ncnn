set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ADD_LIBRARY(interface_face STATIC interface_face.cpp)
add_definitions(-fno-fast-math)
ADD_LIBRARY(SQLITE STATIC ${CMAKE_SOURCE_DIR}/sqlite/sqlite3.c)
#add_subdirectory(${CMAKE_SOURCE_DIR}/sqlite build)
add_executable(demo_face demo.cpp)
target_link_libraries(demo_face interface_face ncnn dlib::dlib SQLITE -ldl)

if(EVALUATE)
    add_executable(detect_face detect_face.cpp)
    target_link_libraries(detect_face ncnn dlib::dlib)    
    add_executable(evaluate evaluate.cpp)
    target_link_libraries(evaluate interface_face ncnn dlib::dlib SQLITE -ldl)
endif()


class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        
        static vector<int> indexes;
        for(int i=0; i<nums.size(); i++)
        {
            for(j = i+1; j<nums.size(); j++)
            {
                if(nums[i] + nums[j] == target)
                {
                    indexes 
                }
            }
        }
    }
};

