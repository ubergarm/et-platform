SW_PLATFORM := ../../../../../../
APP_NAME := fft_kernel
INSTALL_PATH := $(SW_PLATFORM)/build/host-software/$(APP_DIR)/nonMlKernels/
OBJ := ../obj/
SRC := ../src/
INC := -I../ -I../inc -I../inc/dnnlib_subset  -I$(SW_PLATFORM)/build/install-staging/sw-platform-riscv-sysroot/cm-umode/include -I$(SW_PLATFORM)/build/install-staging/sw-platform-riscv-sysroot/include/esperanto
CC := riscv64-unknown-elf-g++
CFLAGS := -mcmodel=medany -DNDEBUG -g3 -O3 -falign-functions=64 -std=gnu++17 -fno-exceptions -fstack-usage
ASFLAGS := -mcmodel=medany
LD := riscv64-unknown-elf-gcc
LDFLAGS := -Tlinker.ld -nostartfiles -nostdlib
LIB := -L$(SW_PLATFORM)/build/install-staging/sw-platform-riscv-sysroot/cm-umode/lib -lcm-umode -lm -lgcc
LD_relink := riscv64-unknown-elf-gcc
LDFLAGS_relink := -Tlinker.relink.ld -nostartfiles -nostdlib
LIB_relink := -L$(SW_PLATFORM)/build/install-staging/sw-platform-riscv-sysroot/cm-umode/lib -lcm-umode -lm -lgcc

all : ../bin/$(APP_NAME).elf ../bin/$(APP_NAME).relink.elf ../bin/$(APP_NAME).S ../bin/$(APP_NAME).relink.S install
clean :
	rm ../obj/*; rm ../bin/*

install:
	 

../bin/$(APP_NAME).elf : ../obj/testOperator_compute.o ../obj/SyncComputeNode.o ../obj/uberKernel_crt.o ../obj/uberKernel.o
	$(LD) $(LDFLAGS) -e _start -o $@ $^ $(LIB)

../bin/$(APP_NAME).relink.elf : ../obj/testOperator_compute.o ../obj/SyncComputeNode.o ../obj/uberKernel_crt.o ../obj/uberKernel.o
	$(LD_relink) $(LDFLAGS_relink) -e _start -o $@ $^ $(LIB)

$(OBJ)%.o: $(SRC)%.cc
	$(CC) $(CFLAGS) ${INC} -c $< -o $@

$(OBJ)%.o: $(SRC)%.S
	$(CC) $(ASFLAGS) ${INC} -c $< -o $@

../bin/$(APP_NAME).S : ../bin/$(APP_NAME).elf
	riscv64-unknown-elf-objdump -lwdS ../bin/$(APP_NAME).elf > ../bin/$(APP_NAME).S
../bin/$(APP_NAME).relink.S : ../bin/$(APP_NAME).relink.elf
	riscv64-unknown-elf-objdump -lwdS ../bin/$(APP_NAME).relink.elf |c++filt > ../bin/$(APP_NAME).relink.S; riscv64-unknown-elf-nm  ../bin/$(APP_NAME).relink.elf --demangle | grep -w "T testOperator_compute"
