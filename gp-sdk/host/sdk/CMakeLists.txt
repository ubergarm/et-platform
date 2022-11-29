
find_package(deviceApi REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(esperantoTrace REQUIRED)

find_package(gflags REQUIRED)


# 
# compile generic_launcher lib
#

add_library(etsoc_gpsdk STATIC src/lib/GenericLauncher.cpp)
target_include_directories(etsoc_gpsdk PUBLIC include)
target_compile_definitions(etsoc_gpsdk PRIVATE $<$<BOOL:${USE_CONAN}>:WITH_CONAN>)
target_link_libraries(etsoc_gpsdk 
    PUBLIC 
        deviceLayer::deviceLayer
        runtime::etrt
    PRIVATE
        esperantoTrace::et_trace
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0.0>>:stdc++fs>
        gflags::gflags
)

# 
# compile specific launcher apps.
#

add_executable(basic_launcher src/basic_launcher/Launcher.cpp)
target_link_libraries(basic_launcher etsoc_gpsdk
                                     gflags::gflags)

add_executable(hello_world_launcher src/hello_world/hello_world.cpp)
target_link_libraries(hello_world_launcher etsoc_gpsdk
                                           gflags::gflags)

add_executable(saxpy_launcher src/saxpy/saxpy.cpp)
target_link_libraries(saxpy_launcher etsoc_gpsdk
                                     gflags::gflags)

add_executable(barrier_launcher src/barrier/barrier.cpp)
target_link_libraries(barrier_launcher etsoc_gpsdk
                                     gflags::gflags)


# FIXME This is WIP. host and device app Share kernel_arguments struct.
# Proposal is follow the "plugin-method" in cmake.
#   device-side exposes the header (INTERFACE mode)  in the executable (with ENABLE_EXPORTS property set)
#   host-side consumes it (either tll or just querying TARGET INTERFACE includes).
# device side is ready, but waiting for Pau to discuss alternatives and conan implicatoins. 
target_include_directories(saxpy_launcher PRIVATE ../../device/tests/saxpy) 
target_include_directories(barrier_launcher PRIVATE ../../device/tests/barrier) 

add_executable(fft_launcher src/fft_launcher/FFTLauncher.cpp)
target_link_libraries(fft_launcher 
    PRIVATE      etsoc_gpsdk
                 gflags::gflags
                  #fftw3f needed for FFT_VERIFICATION 
)
# FIXME. used to get kernel_arguments.h
target_include_directories(fft_launcher PRIVATE ../../device/tests/fftKernel/include) 

install(TARGETS basic_launcher fft_launcher etsoc_gpsdk saxpy_launcher hello_world_launcher
    RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

