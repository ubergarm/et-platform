cmake_minimum_required(VERSION 3.5)


find_package(base REQUIRED)
find_package(deviceApi REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(LLVM REQUIRED)



#
# Compile launcher app
#
add_executable(launcher src/main.cpp)

target_include_directories(launcher
  PUBLIC
    $<BUILD_INTERFACE:${ET_INSTALL_DIR}/sw-platform-x86_64-sysroot/include/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)



#note, we probably don't need deviceApi at this level.
# we can fix depends for experimental fs.. 
#   $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0.0>>:stdc++fs>

target_link_libraries(launcher
  PUBLIC
    deviceApi::deviceApi
    deviceLayer::deviceLayer
    runtime::etrt_static
  PRIVATE
    stdc++fs
)

#getting cmd-line trhough llvm parms lib. This ships by default on our dockers.
target_link_libraries (launcher PRIVATE 
        LLVMSupport)
