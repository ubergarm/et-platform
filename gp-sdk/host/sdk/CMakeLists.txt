cmake_minimum_required(VERSION 3.5)


find_package(base REQUIRED)
find_package(deviceApi REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(LLVM REQUIRED)


# 
# compile generic_launcher app
#

add_executable(generic_launcher src/GenericLauncher.cpp
                                src/Launcher.cpp)

target_link_libraries(generic_launcher
    deviceLayer::deviceLayer
    runtime::etrt_static
    stdc++fs
    LLVMSupport
)


