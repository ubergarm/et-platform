# 
# create abort_manager lib
#

add_library(abort_manager STATIC 
    lib/AbortManager/AbortManager.cpp
    lib/AbortManager/CoreDumper.cpp
)
target_include_directories (abort_manager PUBLIC lib/AbortManager)
target_link_libraries(abort_manager PUBLIC runtime::etrt)

# 
# compile generic_launcher lib
#

add_library(etsoc_gpsdk STATIC 
    src/lib/GenericLauncher.cpp 
    src/lib/RuntimeImpWithCoreDump.cpp
)
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

target_link_libraries(etsoc_gpsdk PUBLIC abort_manager)

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
target_include_directories(saxpy_launcher PRIVATE ../../device/tests/saxpy)
                                   
add_executable(barrier_launcher src/barrier/barrier.cpp)
target_link_libraries(barrier_launcher etsoc_gpsdk
                                     gflags::gflags)


target_include_directories(barrier_launcher PRIVATE ../../device/tests/barrier) 

add_executable(fft_launcher src/fft_launcher/FFTLauncher.cpp)
target_compile_definitions(fft_launcher PRIVATE "FFT_VERIFICATION")
target_link_libraries(fft_launcher 
    PRIVATE      etsoc_gpsdk
                 gflags::gflags
                 FFTW3::fftw3f
)

target_include_directories(fft_launcher PRIVATE ../../device/tests/fftKernel/include) 

add_executable(multiKernel_launcher src/multi_kernel/multiKernel.cpp)
target_link_libraries(multiKernel_launcher etsoc_gpsdk
                                           gflags::gflags)

target_include_directories(multiKernel_launcher PRIVATE ../../device/tests/saxpy)

add_executable(txfma_launcher src/txfma_launcher/txfma_launcher.cpp)
target_link_libraries(txfma_launcher etsoc_gpsdk
                                     gflags::gflags)
target_include_directories(txfma_launcher PRIVATE ../../device/tests/txfma)



                                         
install(TARGETS basic_launcher fft_launcher etsoc_gpsdk saxpy_launcher hello_world_launcher multiKernel_launcher
    RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

