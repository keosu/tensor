CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

PROJECT("Tensor")

set(CMAKE_CXX_FLAGS -g  ) 
set(CMAKE_CXX_FLAGS  -std=c++17) 

include_directories("include") 

add_executable(main "test/main.cpp")   

install(FILES ${CMAKE_SOURCE_DIR}/include/tensor.hpp DESTINATION include)

 
 
