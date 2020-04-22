CC=gcc
CFLAGS=-fPIC -Wall -Wextra
LDFLAGS=-Wall -ljpeg -shared
OBJ=build/libreadjpeg.so
SRC= src/main/cpp/*.cpp

all: $(OBJ)

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(OBJ) $(SRC)

.PHONY: clean

clean:
	rm -f $(OBJ)
