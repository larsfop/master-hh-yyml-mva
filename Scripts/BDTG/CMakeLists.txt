cmake_minimum_required(VERSION 3.9)
project(BDTG CXX)

find_package(ROOT REQUIRED COMPONENTS RIO)

add_executable(bdtg BDTG.C HH_CV_Classification.C HH_CV_Application.C) 
target_link_libraries(bdtg ROOT::Tree ROOT::TMVA)

target_include_directories(bdtg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ROOT_GENERATE_DICTIONARY(BDTG_dict HH_CV_Classification.h HH_CV_Application.h MODULE bdtg LINKDEF LinkDef.h)
