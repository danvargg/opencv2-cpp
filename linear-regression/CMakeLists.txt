cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lregression)

find_package(Torch REQUIRED)

add_executable(lregression linearRegression.cpp CSVReader.h)
target_link_libraries(lregression "${TORCH_LIBRARIES}")
set_property(TARGET lregression PROPERTY CXX_STANDARD 14)
