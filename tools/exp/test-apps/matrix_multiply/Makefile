TARGET = matrix_mul

# CUDA
NVCC = `which nvcc`
ARCH=70

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o matrix_mul -Xptxas -v -arch=sm_$(ARCH) matrix_mul.cu 

test:
	./matrix_mul 

golden:
	./matrix_mul >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ matrix_mul

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
