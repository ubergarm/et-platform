#Description

The device contains the tests to check gpsdk stack including common source code for generated them.

##Building

In order to build follow these steps.

mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<PATH_TO>/Toolchain-gcc-riscv.cmake> -DADDRESS:STRING=<AddressGivenByRuntimeForSpecificTest>
make help
make <desiredTarget>


- Inf: if address is not know you can use a fake @address@!=0 for instance: `-DADDRESS:STRING=0x005050`
