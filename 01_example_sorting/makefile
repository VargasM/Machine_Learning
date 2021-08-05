## =========================================================================
## Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

all: create_sorting_data test_all

create_sorting_data: create_sorting_data.cxx
	g++ -std=c++17 create_sorting_data.cxx -o create_sorting_data

test_all: test_all.cxx sorting/Bubble.h sorting/Insertion.h sorting/Quick.h
	g++ -g -std=c++17 test_all.cxx -o test_all

clean:
	rm -f create_sorting_data test_all *.aux *.log *.pdf

# eof - makefile
