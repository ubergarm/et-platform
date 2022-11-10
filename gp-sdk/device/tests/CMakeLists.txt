#------------------------------------------------------------------------------
# Copyright (C) 2019, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------

find_package(et-common-libs REQUIRED COMPONENTS cm-umode)

include(tests.cmake)

math(EXPR DEBUG_ADDRESS "${ADDRESS} + 0x1000" OUTPUT_FORMAT HEXADECIMAL)
message(STATUS "address used for Debugging-> ${DEBUG_ADDRESS}")

add_etsoc_riscv_executable(print print/print.cc)
add_etsoc_riscv_executable(print2 print2/print2.cc)
