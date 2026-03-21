
set(USE_SANITIZER "" CACHE STRING
    "Define the sanitizer used to build binaries and tests.")

if(USE_SANITIZER)
  # TODO: ensure that the compiler supports these options before adding
  # them.  At the moment, assume that this will just be used with a GNU
  # compatible driver and that the options are spelt correctly in light of that.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")

  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
  endif()


  if(USE_SANITIZER STREQUAL "Address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
  elseif(USE_SANITIZER STREQUAL "Thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
  elseif(USE_SANITIZER STREQUAL "Leaks")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
  elseif(USE_SANITIZER STREQUAL "AddrAndLeaks")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak")
  else()
    message(FATAL_ERROR "unsupported value of USE_SANITIZER: ${USE_SANITIZER}")
  endif()  
endif()
