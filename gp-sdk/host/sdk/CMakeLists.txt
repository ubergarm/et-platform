cmake_minimum_required(VERSION 3.5)


find_package(deviceApi REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(esperantoTrace REQUIRED)

find_package(gflags REQUIRED)



# 
# compile generic_launcher app
#

add_executable(generic_launcher src/GenericLauncher.cpp
                                src/Launcher.cpp)

target_link_libraries(generic_launcher
    deviceLayer::deviceLayer
    runtime::etrt
    esperantoTrace::et_trace
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0.0>>:stdc++fs>
    gflags::gflags
)


