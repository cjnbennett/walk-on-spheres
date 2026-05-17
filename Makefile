# Build script
#
#   make                      # default: h5pcc (MacOS)
#   make CC=cc                # Setonix
#                             # fallback: plain mpicc + explicit HDF5 link
#   make -j4                  # parallel compile, with 4 cores
#   make clean                # remove artefacts

CC        = h5pcc
CFLAGS   ?= -O3 -Wall -Wextra
LDLIBS    = -lm

BUILD_DIR ?= build

TARGET    = $(BUILD_DIR)/poisson
SRCS      = poisson.c bvh.c wos.c mesh.c inside.c npq.c prng.c
OBJS      = $(SRCS:%.c=$(BUILD_DIR)/%.o)
HEADERS   = bvh.h wos.h mesh.h inside.h npq.h grid.h hash.h hdf5_io.h prng.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Any header change rebuilds every object
$(BUILD_DIR)/%.o: %.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
