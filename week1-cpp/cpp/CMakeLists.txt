cmake_minimum_required(VERSION 2.8.12)

PROJECT(week1)
#SET(OpenCV_DIR /home/hp/workfolder/OpenCV-Installation/installation/OpenCV-master/lib/cmake/opencv4)
find_package( OpenCV REQUIRED )

include(../dlib/dlib/cmake )
include_directories(${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()


add_example(facialLandmarkDetector)
add_example(faceAlignment)
add_example(fastWebcamFLD)
add_example(trainFLD)
add_example(webcamStabilizedFLD)
