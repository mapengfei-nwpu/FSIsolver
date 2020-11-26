# Location of the CUDA Toolkit
# CUDA_PATH ?= /usr/local/cuda
CUDA_PATH ?= /usr

# architecture
TARGET_SIZE := 64

# host compiler
HOST_COMPILER := g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -fPIC
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../common/include
LIBRARIES :=

################################################################################
# Gencode arguments
SMS ?= 30 35 37 50 52 60 61 70
# SMS ?= 30

ifeq ($(SMS),)
    $(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif

ifeq ($(GENCODE_FLAGS),)
    # Generate SASS code for each SM architecture listed in $(SMS)
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

    # Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
    HIGHEST_SM := $(lastword $(sort $(SMS)))
    ifneq ($(HIGHEST_SM),)
        GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
    endif
endif

################################################################################

# Target rules
all: build

build: libPolynomialInterpolation.so

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

# In this project, PolynomialInterpolation.cu should be compiled at first.

a.o:PolynomialInterpolation.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

b.o:PolynomialInterpolation.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

libPolynomialInterpolation.so: a.o b.o
	$(EXEC) $(NVCC) -shared -o $@ $+ $(LIBRARIES)
	rm a.o b.o
	mv ./libPolynomialInterpolation.so ../../
	cp ./PolynomialInterpolation.h ../../

clean:
	rm -f a.o b.o

clobber: clean
