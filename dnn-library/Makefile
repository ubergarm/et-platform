BUILD_DIR ?= obj
SRC_DIRS  ?= src
INC_DIRS  := include
LIBNAME   := $(BUILD_DIR)/libLibNodes.a

# TODO: Remove hardcoded paths
INC_DIRS  += ../LibDeviceCommon/include ../fake-fw/include

SRCC := $(shell find $(SRC_DIRS) -name \*.cpp)
OBJC := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCC:.cpp=.o)))
INC_FLAGS := $(addprefix -I, $(INC_DIRS))

CFLAGS := $(INC_FLAGS)
CFLAGS += -g -Wall

ifdef CLANG_BUILD
	CXX    := $(LLVM_RVET)/bin/clang
	CFLAGS += -O3 -mcmodel=medium -target riscv64 -march=rv64-et -mabi=lp64f
	CFLAGS += --sysroot=$(RISCV)/riscv64-unknown-elf
	CFLAGS += -I$(RISCV)/riscv64-unknown-elf/include/c++/8.2.0/
	CFLAGS += -I$(RISCV)/riscv64-unknown-elf/include/c++/8.2.0/riscv64-unknown-elf
else
	CXX    := riscv64-unknown-elf-g++
	CFLAGS += -O2 -mcmodel=medany -march=rv64imf -mabi=lp64f
endif

CXXFLAGS := $(CFLAGS)

all: $(LIBNAME)

# C++ source
$(BUILD_DIR)/%.o: $(SRC_DIRS)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# assembly
$(BUILD_DIR)/%.o: $(SRC_DIRS)/%.S
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) -c $< -o $@

# lib
$(LIBNAME): $(OBJC)
	ar rvs $@ $^

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

MKDIR_P ?= mkdir -p

