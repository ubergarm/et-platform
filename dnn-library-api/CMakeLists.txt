cmake_minimum_required(VERSION 3.17)
project(dnnLibraryApi VERSION 0.7.0 DESCRIPTION "dnnLibrary Host API" LANGUAGES CXX)

option(ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#---------------
# dnnLibrary API library
#---------------

find_package(dnnLibrary REQUIRED)

add_library(dnnLibraryApi
    include/dnnLibraryApi/LibApi.h
    include/dnnLibraryApi/LibTypes.h
    src/Api.cpp
    src/ApiImplSel.cpp
    src/Tensor.cpp
)
add_library(dnnLibraryApi::dnnLibraryApi ALIAS dnnLibraryApi)
target_compile_features(dnnLibraryApi PUBLIC cxx_std_17)
target_compile_options(dnnLibraryApi PRIVATE -Wall $<$<BOOL:${ENABLE_WARNINGS_AS_ERRORS}>:-Werror>)

#dnnLibrary exports some generated header files through an interface library that dnnLibararyApi uses. 
#target_include_directories is used insted of target_link_libraries to cut linking transitivity.
target_include_directories(dnnLibraryApi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        $<TARGET_PROPERTY:dnnLibrary::dnn_lib_host_generated_headers,INTERFACE_INCLUDE_DIRECTORIES>
)

set_target_properties(dnnLibraryApi PROPERTIES POSITION_INDEPENDENT_CODE ON)

#-------------------
# Configure package
#-------------------

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/dnnLibraryApiConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryApiConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnnLibraryApi
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryApiConfigVersion.cmake
    VERSION ${VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryApiConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dnnLibraryApiConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnnLibraryApi
)



#-----------------
# Install targets
#-----------------

install(TARGETS dnnLibraryApi
    EXPORT dnnLibraryApiTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT dnnLibraryApiTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnnLibraryApi
    NAMESPACE dnnLibraryApi::
)

