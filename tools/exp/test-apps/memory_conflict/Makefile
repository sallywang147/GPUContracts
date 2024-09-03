TARGET = add_with_bank_conflict

# CUDA
NVCC = `which nvcc`
ARCH=70

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -arch=sm_$(ARCH) -Xptxas -v -cubin add_with_bank_conflict.cu -o add_with_bank_conflict.cubin cuobjdump --dump-sass add_with_bank_conflict.cubin
	$(NVCC) -o add_with_bank_conflict -Xptxas -v -arch=sm_$(ARCH) add_with_bank_conflict.cu 

test:
	./$(TARGET) 

golden:
	./$(TARGET) >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
TARGET = add_with_bank_conflict

# CUDA
NVCC = $(shell which nvcc)  # Correctly use $(shell ...) to get the path of nvcc
ARCH = 35

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o $(TARGET) -Xptxas -v -arch=sm_$(ARCH) add_with_bank_conflict.cu 

test:
	./$(TARGET) 

golden:
	./$(TARGET) >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
