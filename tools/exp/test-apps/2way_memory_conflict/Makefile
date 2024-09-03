TARGET = bank_conflict

# CUDA
NVCC = `which nvcc`
ARCH=70

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o bank_conflict -Xptxas -v -arch=sm_$(ARCH) bank_conflict.cu 

test:
	./bank_conflict 

golden:
	./bank_conflict >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ bank_conflict

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
