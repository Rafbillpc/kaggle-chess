EXE := Obsidian
EVALFILE = nn.net

M64     = -m64 -mpopcnt
MSSE2   = $(M64) -msse -msse2
MSSSE3  = $(MSSE2) -mssse3
MAVX2   = $(MSSSE3) -msse4.1 -mbmi -mfma -mavx2

FILES = src/bitboard.cpp \
	src/cuckoo.cpp \
	src/evaluate.cpp \
	src/main.cpp \
	src/movegen.cpp \
	src/movepick.cpp \
	src/nnue.cpp \
	src/position.cpp \
	src/search.cpp \
	src/threads.cpp \
	src/timeman.cpp \
	src/tt.cpp \
	src/tuning.cpp \
	src/uci.cpp \
	src/ucioption.cpp \
	src/zobrist.cpp \
	src/nnue_load.cpp

OBJS = $(FILES:.cpp=.o)
OBJS := $(OBJS:.c=.o)

OPTIMIZE = -Os -fno-stack-protector -fno-math-errno -fno-jump-tables -funroll-loops -fno-rtti -fno-exceptions -flto

FLAGS = -std=c++23 -DNDEBUG -DEvalFile=\"$(EVALFILE)\" $(OPTIMIZE)
FLAGS += -Wno-parentheses -Wno-deprecated-enum-enum-conversion -Wno-deprecated-volatile
# FLAGS += -g

# FLAGS += -DSAVESAMPLES

END_FLAGS :=

ifdef quant
	FLAGS += -DUSE_QUANT
	EVALFILE = "nn_quant.net"
endif

ifdef shared
	FLAGS += -fvisibility=hidden -DSHARED=1 -nostdlib -fno-use-cxa-atexit -ffreestanding
  # FLAGS += -fvisibility=hidden -DSHARED=1 -fPIC
	END_FLAGS += -shared
	EXE = Obsidian.so
endif

FLAGS += $(MAVX2)
FLAGS += -DUSE_PEXT -mbmi2

# src/movegen.o: FLAGS += -O2
# src/movepick.o: FLAGS += -O2
# src/position.o: FLAGS += -O2
src/search.o: FLAGS += -O2
src/evaluate.o: FLAGS += -O2
src/nnue.o: FLAGS += -O2

%.o: %.cpp
	clang++ $(FLAGS) -c $< -o $@

nopgo: $(OBJS)
	clang++ $(END_FLAGS) $(FLAGS) $(OBJS) -o $(EXE)

clean:
	rm -f $(OBJS)
