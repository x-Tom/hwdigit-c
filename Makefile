neural_net:
	gcc -Wall -g -Ofast main.c neural_net.c math_functions.c display.c aux.c -o neural_net -I include -L lib -l SDL2-2.0.0 -lSDL2_ttf

