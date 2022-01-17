ARCH = 75
KERNEL = ld_global_cache

all:
	python3 -m turingas.main -i $(KERNEL).sass -o $(KERNEL).cubin -arch $(ARCH)
	# nvcc main.cu -w -lcuda -arch=sm_$(ARCH) -o $(KERNEL).out
	# nvcc $(KERNEL).cu -w -lcuda -arch=sm_$(ARCH) -o pre_$(KERNEL).out
	# cuobjdump -sass pre_$(KERNEL).out > pre_$(KERNEL).sass
	# python3 ~/Develop/turingas/tools/disasm.py pre_$(KERNEL).out > human_pre_$(KERNEL).sass
clean:
	rm $(KERNEL).cubin $(KERNEL) pre_$(KERNEL)
