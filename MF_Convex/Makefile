CC=g++

cpps=$(shell ls *.cpp)
hfiles=$(shell ls *.h)
exec='MF_Convex'

all: 
	$(CC) -O3 -std=c++0x MF_Convex.cpp  -o $(exec)

push:
	git add $(cpps) $(hfiles) Makefile README.md
	git commit -m "upgrade $(exec)"
	git push

test: 
	$(CC) -fopenmp -O3 test_CSW.cpp -o test_CSW


clean:
	rm -rf $(exec) test_CSW
