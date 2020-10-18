



LIBRARIES = libparticles.so

particles.o: particles.cpp
	$(EXEC) g++ -o $@ -c $<

particles: particles.o
	$(EXEC) g++ -o $@ $+ $(LIBRARIES)

clean:
	rm -f particles particles.o

