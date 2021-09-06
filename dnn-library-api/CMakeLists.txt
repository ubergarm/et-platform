cmake_minimum_required(VERSION 3.17)
project(dnnLibraryApi
    VERSION 0.1.0
    DESCRIPTION "dnnLibrary Host API"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#---------------
# dnnLibrary API library
#---------------

set(CMAKE_CXX_STANDARD 17)

add_library(dnnLibraryApi INTERFACE)
add_library(dnnLibraryApi::dnnLibraryApi ALIAS dnnLibraryApi)
target_include_directories(dnnLibraryApi
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

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

