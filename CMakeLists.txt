cmake_minimum_required(VERSION 3.10)
project(AlmgrenChriss CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release flags: optimise and enable all warnings.
if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O2 -Wall -Wextra)
endif()

add_executable(almgren_chriss src/almgren_chriss.cpp)
