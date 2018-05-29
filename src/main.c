#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "SDL.h"
#include "SDL_ttf.h"

#define SCREEN_W 1200
#define SCREEN_H 800

/*
 * TODO(bkaylor): Delaunay implementation- euclidean
 * TODO(bkaylor): Write diagram to file
 * TODO(bkaylor): Variable window size
 * TODO(bkaylor): Single texture?
 * TODO(bkaylor): "WORKING ..." message
 * TODO(bkaylor): More distance functions?
 * TODO(bkaylor): ERROR: Stops working for high point counts. Find max/fix?
*/

enum Distance_Formula
{
	EUCLIDEAN,
	MANHATTAN
};

typedef struct Point_Struct
{
	int x;
	int y;
	int r;
	int g;
	int b;
} Point;

void voronoi(SDL_Renderer *, int, enum Distance_Formula);
void fast(SDL_Renderer *, int, enum Distance_Formula);

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

	SDL_Event event;
	int quit = 0, doiter = 0;
	int frame = 0;
	char frame_s[10];
	char type_s[10];
	char point_s[10];
    int show_text = 1;
    int fast_mode = 0;

	enum Distance_Formula dist_type = EUCLIDEAN;
	int pointc = 20;

	// Setup
	printf("\n");
	srand((unsigned) time(NULL));
	unsigned int start_time, end_time, frame_time;
	frame_time = 0;
    sprintf(type_s, "Euclidean");

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

							case SDLK_k:
								if (dist_type == MANHATTAN)
								{
									dist_type = EUCLIDEAN;
                                    sprintf(type_s, "Euclidean");
								}
								else
								{
									dist_type = MANHATTAN;
                                    sprintf(type_s, "Manhattan");
								}
								break;

							case SDLK_i:
								pointc = pointc + 10;
								break;

							case SDLK_o:
								if (pointc > 10)
								{
									pointc = pointc - 10;
								}
								break;

							case SDLK_h:
                                if (show_text == 1) {
                                    show_text = 0;
                                } else if (show_text == 0) {
                                    show_text = 1;
                                }
								break;

                            case SDLK_f:
                                if (fast_mode == 1) {
                                    fast_mode = 0;
                                } else if (fast_mode == 0) {
                                    fast_mode = 1;
                                }
                                break;

							case SDLK_ESCAPE:
                                quit = 1;
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

        if (!quit) {
            doiter = 0;

            // Render
            SDL_RenderClear(ren);

            start_time = SDL_GetTicks();
            if (fast_mode) {
                fast(ren, pointc, dist_type);
            } else {
                voronoi(ren, pointc, dist_type);
            }
            end_time = SDL_GetTicks();

            frame_time = end_time - start_time;
            sprintf(frame_s, "%u ms", frame_time);

            sprintf(point_s, "%d points", pointc);

            SDL_Surface *frame_surface = TTF_RenderText_Solid(font, frame_s, font_color);
            SDL_Texture *frame_texture = SDL_CreateTextureFromSurface(ren, frame_surface);
            int frame_x, frame_y;
            SDL_QueryTexture(frame_texture, NULL, NULL, &frame_x, &frame_y);
            SDL_Rect frame_rect = {5 , 5, frame_x, frame_y};

            if (show_text) {
                SDL_RenderCopy(ren, frame_texture, NULL, &frame_rect);
            }

            SDL_Surface *type_surface = TTF_RenderText_Solid(font, type_s, font_color);
            SDL_Texture *type_texture = SDL_CreateTextureFromSurface(ren, type_surface);
            int type_x, type_y;
            SDL_QueryTexture(type_texture, NULL, NULL, &type_x, &type_y);
            SDL_Rect type_rect = {5 , 5 + 10, type_x, type_y};

            if (show_text) {
                SDL_RenderCopy(ren, type_texture, NULL, &type_rect);
            }

            SDL_Surface *point_surface = TTF_RenderText_Solid(font, point_s, font_color);
            SDL_Texture *point_texture = SDL_CreateTextureFromSurface(ren, point_surface);
            int point_x, point_y;
            SDL_QueryTexture(point_texture, NULL, NULL, &point_x, &point_y);
            SDL_Rect point_rect = {5 , 5 + 20, point_x, point_y};

            if (show_text) {
                SDL_RenderCopy(ren, point_texture, NULL, &point_rect);
            }

            SDL_RenderPresent(ren);
            ++frame;

            // Cleanup
            SDL_FreeSurface(frame_surface);
            SDL_DestroyTexture(frame_texture);

            SDL_FreeSurface(type_surface);
            SDL_DestroyTexture(type_texture);

            SDL_FreeSurface(point_surface);
            SDL_DestroyTexture(point_texture);
        }
	}

	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}

void fast(SDL_Renderer *ren, int pointc, enum Distance_Formula dist_type)
{
	// Assign random points
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
            SDL_SetRenderDrawColor(ren, rand() % 255, rand() % 255, rand() % 255, 255);
            SDL_RenderDrawPoint(ren, i, j);
        }
    }
}

void voronoi(SDL_Renderer *ren, int pointc, enum Distance_Formula dist_type)
{
	// Assign random points
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
            float x_distance, y_distance;
			for (int k = 0; k < pointc; ++k)
			{
				// Get distance from each point
				switch(dist_type)
				{
					case (MANHATTAN):
						distances[k] = abs(points[k].x - i) + abs(points[k].y - j);
						break;
					case (EUCLIDEAN):
					default:
                        x_distance = abs(points[k].x - i);
                        y_distance = abs(points[k].y - j);
						distances[k] = (x_distance * x_distance) + (y_distance * y_distance);
						break;

				}
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
