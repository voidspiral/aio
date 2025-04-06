CC = mpicc
CFLAGS = -fPIC -Wall -g
LDFLAGS = -shared -ldl

TARGET = libmpiio.so
SRCS = mpiio.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean