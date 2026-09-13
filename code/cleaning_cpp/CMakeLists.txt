cmake_minimum_required(VERSION 3.20)
project(cleaning_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(cleaning_cpp
  src/sha256.cpp
  src/xlsx.cpp
  src/panel.cpp
  src/rules.cpp
  src/writers.cpp
  src/main.cpp
)

target_include_directories(cleaning_cpp PRIVATE include)
target_compile_options(cleaning_cpp PRIVATE -O2 -Wall -Wextra)

find_package(ZLIB REQUIRED)
find_library(TINYXML2_LIB tinyxml2 REQUIRED)
find_path(TINYXML2_INC tinyxml2.h REQUIRED)
target_include_directories(cleaning_cpp PRIVATE ${TINYXML2_INC})
target_link_libraries(cleaning_cpp PRIVATE ZLIB::ZLIB ${TINYXML2_LIB})
