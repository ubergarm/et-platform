

# 
# compile hello-world launcher
#

add_executable(my_hello_world hello_world/hello_world.cpp)
target_link_libraries(my_hello_world  etsoc_gpsdk
                                      gflags::gflags)
