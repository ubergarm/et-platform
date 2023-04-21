
set(DEVICE_KERNELS_DIR ${PROJECT_SOURCE_DIR}/../device/tests)

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
target_include_directories(etsoc_gpsdk PUBLIC include sdk/include)
target_link_libraries(etsoc_gpsdk 
    PUBLIC 
        deviceLayer::deviceLayer
        runtime::etrt
        abort_manager
    PRIVATE
        esperantoTrace::et_trace
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0.0>>:stdc++fs>
)

# 
# compile specific launcher apps.
#

if (BUILD_TESTS)

add_executable(basic_launcher src/basic_launcher/Launcher.cpp)

target_link_libraries(basic_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(hello_world_launcher src/hello_world/hello_world.cpp)
target_link_libraries(hello_world_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(saxpy_launcher src/saxpy/saxpy.cpp)
target_include_directories(saxpy_launcher PRIVATE ${DEVICE_KERNELS_DIR}/saxpy)
target_link_libraries(saxpy_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(barrier_launcher src/barrier/barrier.cpp)
target_include_directories(barrier_launcher PRIVATE ${DEVICE_KERNELS_DIR}/barrier) 
target_link_libraries(barrier_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(fft_launcher src/fft_launcher/FFTLauncher.cpp)
target_include_directories(fft_launcher PRIVATE ${DEVICE_KERNELS_DIR}/fftKernel/include)
target_compile_definitions(fft_launcher PRIVATE FFT_VERIFICATION)
target_link_libraries(fft_launcher 
    PRIVATE
        etsoc_gpsdk
        FFTW3::fftw3f
)

add_executable(multiKernel_launcher src/multi_kernel/multiKernel.cpp)
target_include_directories(multiKernel_launcher PRIVATE ${DEVICE_KERNELS_DIR}/saxpy)
target_link_libraries(multiKernel_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(txfma_launcher src/txfma_launcher/txfma_launcher.cpp)
target_include_directories(txfma_launcher PRIVATE ${DEVICE_KERNELS_DIR}/txfma)
target_link_libraries(txfma_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(multiDevice_launcher src/multi_device/multiDevice.cpp)
target_include_directories(multiDevice_launcher PRIVATE ${DEVICE_KERNELS_DIR}/saxpy)
target_link_libraries(multiDevice_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(multiDeviceHang_launcher src/multi_device/multiDeviceHang.cpp)
target_link_libraries(multiDeviceHang_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(trapping_launcher src/trapping/trapping.cpp)
target_link_libraries(trapping_launcher
    PRIVATE
        etsoc_gpsdk
)

add_executable(matmul_launcher src/autogen_matmul/autogen_matmul.cpp)
target_include_directories(matmul_launcher PRIVATE ${DEVICE_KERNELS_DIR}/autogenMatmul)
target_link_libraries(matmul_launcher
    PRIVATE
        etsoc_gpsdk
)
      
install(TARGETS basic_launcher fft_launcher etsoc_gpsdk saxpy_launcher hello_world_launcher multiKernel_launcher multiDevice_launcher multiDeviceHang_launcher trapping_launcher
    RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

endif()
