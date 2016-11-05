@pushd bin 
gcc -Wall -g -std=c99 -I"../include" -c ../src/main.c
gcc main.o -L"../lib" -Wl,-subsystem,windows -lmingw32 -lSDL2main -lSDL2 -lSDL2_ttf -o voronoi
@popd
