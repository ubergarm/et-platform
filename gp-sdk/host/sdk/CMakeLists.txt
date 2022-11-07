
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


add_executable(fft_launcher src/fft_launcher/FFTLauncher.cpp)
target_link_libraries(fft_launcher etsoc_gpsdk
                               gflags::gflags)

install(TARGETS basic_launcher fft_launcher etsoc_gpsdk
    RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

