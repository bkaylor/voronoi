#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "SDL.h"
#include "SDL_ttf.h"

#define SCREEN_W 1200
#define SCREEN_H 800

typedef struct Point_Struct
{
	int x;
	int y;
	int r;
	int g;
	int b;
} Point;

void voronoi(SDL_Renderer *, int);

int main(int argc, char *argv[])
{
	SDL_Init(SDL_INIT_EVERYTHING);

	// Setup window
	SDL_Window *win = SDL_CreateWindow("Voronoi",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
			SCREEN_W, SCREEN_H,
			SDL_WINDOW_SHOWN);

	// Setup renderer
	SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	// Setup font
	TTF_Init();
	TTF_Font *font = TTF_OpenFont("liberation.ttf", 12);
	if (!font)
	{
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error: Font", TTF_GetError(), win);
		return -666;
	}

	SDL_Color font_color = {255, 255, 255};
	SDL_Rect message_rect = {0, 0, 80, 30};

	SDL_Event event;
	int quit = 0, doiter = 0;
	int frame = 0;
	char frame_s[10];

	// Setup
	printf("\n");
	srand((unsigned) time(NULL));
	unsigned int start_time, end_time, frame_time;
	frame_time = 0;

	// Main Loop
	while (!quit)
	{

		while (!doiter && !quit)
		{
			// Input
			while (SDL_PollEvent(&event))
			{
				switch (event.type)
				{
					case SDL_KEYDOWN:
						switch (event.key.keysym.sym)
						{
							case SDLK_SPACE:
								doiter = 1;
								break;
						}
						break;
					case SDL_QUIT:
						quit = 1;
						break;
					default:
						break;
				}
			}
		}

		doiter = 0;
		start_time = SDL_GetTicks();

		// Simulate
		sprintf(frame_s, "%u ms", frame_time);
		SDL_Surface *message_surf = TTF_RenderText_Solid(font, frame_s, font_color);
		SDL_Texture *message_txtr = SDL_CreateTextureFromSurface(ren, message_surf);

		// Render
		SDL_RenderClear(ren);
		voronoi(ren, 20);

		SDL_RenderCopy(ren, message_txtr, NULL, &message_rect);
		SDL_RenderPresent(ren);
		++frame;

		// Cleanup
		SDL_FreeSurface(message_surf);
		SDL_DestroyTexture(message_txtr);
		end_time = SDL_GetTicks();
		frame_time = end_time - start_time;
	}

	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}

void voronoi(SDL_Renderer *ren, int pointc)
{
	// Assign 20 random points
	Point *points = malloc(sizeof(Point) * pointc);

	for (int i = 0; i < pointc; ++i)
	{
		points[i].x = rand() % SCREEN_W;
		points[i].y = rand() % SCREEN_H;

		points[i].r = rand() % 255;
		points[i].g = rand() % 255;
		points[i].b = rand() % 255;
	}
	
	for (int i = 0; i < SCREEN_W; ++i)
	{
		for (int j = 0; j < SCREEN_H; ++j)
		{
			float distances[pointc];
			for (int k = 0; k < pointc; ++k)
			{
				// Get distance from each point
				distances[k] = sqrt(abs(points[k].x - i) + abs(points[k].y - j));
			}

			// Assign color based on closest
			int minind = 0;
			for (int k = 0; k < pointc; ++k)
			{
				if (distances[k] < distances[minind])
				{
					minind = k;
				}

				// Color the points themselves black
				if (distances[minind] < 2.0)
				{
					SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
				}
				else
				{
					SDL_SetRenderDrawColor(ren, points[minind].r, points[minind].b, points[minind].g, 255);
				}
			}

			// Draw pixel! :^)
			SDL_RenderDrawPoint(ren, i, j);
		}
	}
}
