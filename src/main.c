
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include "SDL.h"
#include "SDL_ttf.h"

#include "shared_with_cuda.h"

/*
 * TODO(bkaylor): Delaunay implementation- euclidean
 * TODO(bkaylor): Write diagram to file
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

typedef struct Color_Struct
{
    int r;
    int g;
    int b;
} Color;

typedef struct Circle_Struct
{
    Point center;
    float radius;
} Circle;

typedef struct Triangle_Struct
{
    Point points[3];
    Circle circumcircle;
    bool evicted;
} Triangle;

typedef struct Edge_Struct
{
    Point points[2];
} Edge;

typedef struct
{
    Point point;
    float slope;
    int b;
} Line;

extern void voronoi_cuda(Point *, char *);

/*
void render_triangulation(SDL_Renderer *ren, Triangle *triangulation, int triangle_count)
{
    // Draw a wireframe of the triangulation.
    SDL_SetRenderDrawColor(ren, foreground.r, foreground.g, foreground.b, 255);
    SDL_RenderClear(ren);
    SDL_SetRenderDrawColor(ren, background.r, background.g, background.b, 255);

    for (int i = 0; i < triangle_count; i += 1)
    {
        Triangle t = triangulation[i];
        if (t.evicted) continue;

        Point a, b, c;
        a = t.points[0]; 
        b = t.points[1]; 
        c = t.points[2];

        SDL_RenderDrawLine(ren, a.x, a.y, b.x, b.y);
        SDL_RenderDrawLine(ren, b.x, b.y, c.x, c.y);
        SDL_RenderDrawLine(ren, c.x, c.y, a.x, a.y);
    }
}
*/

void draw_circle(SDL_Renderer *renderer, int32_t centreX, int32_t centreY, int32_t radius)
{
   const int32_t diameter = (radius * 2);

   int32_t x = (radius - 1);
   int32_t y = 0;
   int32_t tx = 1;
   int32_t ty = 1;
   int32_t error = (tx - diameter);

   while (x >= y)
   {
      //  Each of the following renders an octant of the circle
      SDL_RenderDrawPoint(renderer, centreX + x, centreY - y);
      SDL_RenderDrawPoint(renderer, centreX + x, centreY + y);
      SDL_RenderDrawPoint(renderer, centreX - x, centreY - y);
      SDL_RenderDrawPoint(renderer, centreX - x, centreY + y);
      SDL_RenderDrawPoint(renderer, centreX + y, centreY - x);
      SDL_RenderDrawPoint(renderer, centreX + y, centreY + x);
      SDL_RenderDrawPoint(renderer, centreX - y, centreY - x);
      SDL_RenderDrawPoint(renderer, centreX - y, centreY + x);

      if (error <= 0)
      {
         ++y;
         error += ty;
         ty += 2;
      }

      if (error > 0)
      {
         --x;
         tx += 2;
         error += (tx - diameter);
      }
   }
}

/*
   TODO(bryan): Implement delaunay triangulation algorithm pseudocode

   function BowyerWatson (pointList)
      // pointList is a set of coordinates defining the points to be triangulated
      triangulation := empty triangle mesh data structure
      add super-triangle to triangulation // must be large enough to completely contain all the points in pointList
      for each point in pointList do // add all the points one at a time to the triangulation
         badTriangles := empty set
         for each triangle in triangulation do // first find all the triangles that are no longer valid due to the insertion
            if point is inside circumcircle of triangle
               add triangle to badTriangles
         polygon := empty set
         for each triangle in badTriangles do // find the boundary of the polygonal hole
            for each edge in triangle do
               if edge is not shared by any other triangles in badTriangles
                  add edge to polygon
         for each triangle in badTriangles do // remove them from the data structure
            remove triangle from triangulation
         for each edge in polygon do // re-triangulate the polygonal hole
            newTri := form a triangle from edge to point
            add newTri to triangulation
      for each triangle in triangulation // done inserting points, now clean up
         if triangle contains a vertex from original super-triangle
            remove triangle from triangulation
      return triangulation
*/

// TODO(bkaylor): What should these actually be?
#define MAX_POINTS 1000
#define MAX_TRIANGLES MAX_POINTS*3
#define MAX_EDGES MAX_TRIANGLES*100

int are_equivalent_points(Point a, Point b)
{
    return (a.x == b.x) && (a.y == b.y);
}

int are_equivalent_edges(Edge a, Edge b)
{
    return (are_equivalent_points(a.points[0], b.points[0]) && are_equivalent_points(a.points[1], b.points[1])) ||
           (are_equivalent_points(a.points[1], b.points[0]) && are_equivalent_points(a.points[0], b.points[1]));
}

Circle get_circumcircle_of_triangle(Triangle triangle)
{
        Line bisector_01;
        bisector_01.point.x = (triangle.points[0].x + triangle.points[1].x)/2;
        bisector_01.point.y = (triangle.points[0].y + triangle.points[1].y)/2;

        if ((triangle.points[1].y - triangle.points[0].y) == 0)
        {
            bisector_01.slope = (float)INT_MAX;
        }
        else
        {
            bisector_01.slope = -(float)(triangle.points[1].x - triangle.points[0].x) / (float)(triangle.points[1].y - triangle.points[0].y);
        }
        bisector_01.b = (int)(bisector_01.point.y - (bisector_01.slope * bisector_01.point.x));

        Line bisector_12;
        bisector_12.point.x = (triangle.points[1].x + triangle.points[2].x)/2;
        bisector_12.point.y = (triangle.points[1].y + triangle.points[2].y)/2;

        if ((triangle.points[2].y - triangle.points[1].y) == 0)
        {
            bisector_12.slope = (float)INT_MAX;
        }
        else
        {
            bisector_12.slope = -(float)(triangle.points[2].x - triangle.points[1].x) / (float)(triangle.points[2].y - triangle.points[1].y);
        }
        bisector_12.b = (int)(bisector_12.point.y - (bisector_12.slope * bisector_12.point.x));

        /*
        y = m1x + b1
        y = m2x + b2
        x = (y - b2)/m2

        y = m1 ((y - b2) / m2) + b1

        m1x + b1 = m2x + b2
        x = (b1 - b2) / (m2 - m1)

        y = m1
        */

        Point center;
        center.x = (int)((bisector_01.b - bisector_12.b) / (bisector_12.slope - bisector_01.slope));
        center.y = (int)((bisector_01.slope * center.x) + bisector_01.b);

        // Get the distance from the circumcenter to one of the triangle's points.
        float x_distance = (float)abs(center.x - triangle.points[0].x);
        float y_distance = (float)abs(center.y - triangle.points[0].y);

        Circle circle;
        circle.center = center;
        circle.radius = sqrtf(x_distance*x_distance + y_distance*y_distance);

        return circle;

}

// TODO(bkaylor): Occasional crash during a pass on fast mode, before it prints or draws anything.
//                Seems to happen more when there are a relatively small amount of points.
void fast(SDL_Renderer *ren, int point_count, int window_w, int window_h)
{
    Color foreground;
    foreground.r = (rand() %  128) + 128;
    foreground.g = (rand() %  128) + 128;
    foreground.b = (rand() %  128) + 128;

    Color background;
    background.r = rand() %  128;
    background.g = rand() %  128;
    background.b = rand() %  128;

	// Assign random points
	Point points[MAX_POINTS];

	for (int i = 0; i < point_count; ++i)
	{
		points[i].x = rand() % window_w;
		points[i].y = rand() % window_h;

		points[i].r = rand() % 255;
		points[i].g = rand() % 255;
		points[i].b = rand() % 255;
    }

    Triangle triangulation[MAX_TRIANGLES];

    // Add supertriangle 
    Point supertriangle_lower_left = {-3 * window_w, -3 * window_h};
    Point supertriangle_upper_middle = {window_w/3, 3*window_h};
    Point supertriangle_lower_right = {3 * window_w, -3*window_h};
    triangulation[0].points[0] = supertriangle_lower_left;
    triangulation[0].points[1] = supertriangle_upper_middle;
    triangulation[0].points[2] = supertriangle_lower_right;
    triangulation[0].evicted = false;

    int triangle_count = 1;
    Triangle bad_triangles[MAX_TRIANGLES];

    for (int i = 0; i < point_count; i++)
    {
        // Add this point to the triangulation.
        int bad_triangle_count = 0;

        // Find the bad triangles to evict from triangulation (triangles that contain this point).
        for (int j = 0; j < triangle_count; ++j)
        {
            if (triangulation[j].evicted) continue;

            // Is point inside triangle's circumcircle?

            // The circumcenter of a triangle is not just an average of the 3 points.
            // You have to get the intersection of the perpindicular bisectors of two sides.
            triangulation[j].circumcircle = get_circumcircle_of_triangle(triangulation[j]);

            // Check if point inside    
            float x_distance = (float)abs(triangulation[j].circumcircle.center.x - points[i].x);
            float y_distance = (float)abs(triangulation[j].circumcircle.center.y - points[i].y);

            if (sqrt(x_distance*x_distance + y_distance*y_distance) < triangulation[j].circumcircle.radius) {
                bad_triangles[bad_triangle_count] = triangulation[j];
                bad_triangle_count += 1;

                triangulation[j].evicted = true;
            }
        }

        if (bad_triangle_count > 0)
        {
            Edge *polygon = malloc(sizeof(Edge) * MAX_EDGES);
            int polygon_edge_count = 0;

            // Add any unique edges from the bad triangles to the polygon.
            for (int j = 0; j < bad_triangle_count; j++)
            {
                Edge a, b, c;
                a.points[0] = bad_triangles[j].points[0];
                a.points[1] = bad_triangles[j].points[1];
                b.points[0] = bad_triangles[j].points[1];
                b.points[1] = bad_triangles[j].points[2];
                c.points[0] = bad_triangles[j].points[2];
                c.points[1] = bad_triangles[j].points[0];

                // TODO(bkaylor): Roll this into three loops, one per edge?
                bool is_a_unique = true;
                bool is_b_unique = true;
                bool is_c_unique = true;
                for (int k = 0; k < bad_triangle_count; k++)
                {
                    // TODO(bkaylor): Still not sure if this should be here. Probably.
                    if (k == j) continue;

                    Edge d, e, f;
                    d.points[0] = bad_triangles[k].points[0];
                    d.points[1] = bad_triangles[k].points[1];
                    e.points[0] = bad_triangles[k].points[1];
                    e.points[1] = bad_triangles[k].points[2];
                    f.points[0] = bad_triangles[k].points[2];
                    f.points[1] = bad_triangles[k].points[0];

                    if (are_equivalent_edges(a, d) || are_equivalent_edges(a, e) || are_equivalent_edges(a, f)) {
                        is_a_unique = false;
                    }

                    if (are_equivalent_edges(b, d) || are_equivalent_edges(b, e) || are_equivalent_edges(b, f)) {
                        is_b_unique = false;
                    }

                    if (are_equivalent_edges(c, d) || are_equivalent_edges(c, e) || are_equivalent_edges(c, f)) {
                        is_c_unique = false;
                    }
                }

                if (is_a_unique)
                {
                    polygon[polygon_edge_count] = a;
                    polygon_edge_count += 1;
                }

                if (is_b_unique)
                {
                    polygon[polygon_edge_count] = b;
                    polygon_edge_count += 1;
                }

                if (is_c_unique)
                {
                    polygon[polygon_edge_count] = c;
                    polygon_edge_count += 1;
                }
            }

            // Add a triangle to the triangulation for each edge on the polygon.
            for (int j = 0; j < polygon_edge_count; j += 1)
            {
                Edge edge = polygon[j];

                Triangle new_triangle;
                new_triangle.points[0] = points[i];
                new_triangle.points[1] = edge.points[0];
                new_triangle.points[2] = edge.points[1];
                new_triangle.evicted = false;

                triangulation[triangle_count] = new_triangle;
                triangle_count += 1;
            }

            // printf("For point #%d, added %d triangles.\n", i, polygon_edge_count);
#if 0
            // Intermediate drawing for debugging
            if (polygon_edge_count > 0)
            {
                // Draw a wireframe of the triangulation.
                SDL_SetRenderDrawColor(ren, foreground.r, foreground.g, foreground.b, 255);
                SDL_RenderClear(ren);
                SDL_SetRenderDrawColor(ren, background.r, background.g, background.b, 255);

                for (int i = 0; i < triangle_count; i += 1)
                {
                    Triangle t = triangulation[i];
                    if (t.evicted) continue;

                    Point a, b, c;
                    a = t.points[0]; 
                    b = t.points[1]; 
                    c = t.points[2];

                    SDL_RenderDrawLine(ren, a.x, a.y, b.x, b.y);
                    SDL_RenderDrawLine(ren, b.x, b.y, c.x, c.y);
                    SDL_RenderDrawLine(ren, c.x, c.y, a.x, a.y);
                    SDL_RenderPresent(ren);
                    // SDL_Delay(500/triangle_count);
                }
            }
#endif
        free(polygon);
        }
    }

    // Remove any triangles from the triangulation if they use a supertriangle vertex.
    for (int j = 0; j < triangle_count; j += 1)
    {
        if (
                (are_equivalent_points(triangulation[j].points[0], triangulation[0].points[0]) ||
                 are_equivalent_points(triangulation[j].points[0], triangulation[0].points[1]) ||
                 are_equivalent_points(triangulation[j].points[0], triangulation[0].points[2]))
                ||
                (are_equivalent_points(triangulation[j].points[1], triangulation[0].points[0]) ||
                 are_equivalent_points(triangulation[j].points[1], triangulation[0].points[1]) ||
                 are_equivalent_points(triangulation[j].points[1], triangulation[0].points[2]))
                ||
                (are_equivalent_points(triangulation[j].points[2], triangulation[0].points[0]) ||
                 are_equivalent_points(triangulation[j].points[2], triangulation[0].points[1]) ||
                 are_equivalent_points(triangulation[j].points[2], triangulation[0].points[2]))
            )
        {
            triangulation[j].evicted = true;
        }
    }

    // Print out triangles.
#if 0
    for (int i = 0; i < triangle_count; i += 1)
    {
        Triangle t = triangulation[i];
        if (t.evicted) continue;

        printf("(%d, %d) ", t.points[0].x, t.points[0].y);
        printf("(%d, %d) ", t.points[1].x, t.points[1].y);
        printf("(%d, %d) ", t.points[2].x, t.points[2].y);
        printf("(%d, %d, %f) ", t.circumcircle.center.x, t.circumcircle.center.y, t.circumcircle.radius);
        printf("\n");
    }
#endif

    // Draw a wireframe of the triangulation.
    SDL_SetRenderDrawColor(ren, (Uint8)background.r, (Uint8)background.g, (Uint8)background.b, (Uint8)255);
    SDL_RenderClear(ren);
    // SDL_SetRenderDrawColor(ren, foreground.r, foreground.g, foreground.b, 255);

    // Draw circumcircles.
    SDL_SetRenderDrawColor(ren, (Uint8)(background.r + 15), (Uint8)(background.g + 15), (Uint8)(background.b + 15), 255);
    for (int i = 0; i < triangle_count; i += 1)
    {
        Triangle t = triangulation[i];
        if (t.evicted) continue;
        // TODO(bkaylor): There should be a real way to resolve this earlier, where the center is computed..
        if ((t.circumcircle.center.x == INT_MAX || t.circumcircle.center.x == INT_MIN) || 
            (t.circumcircle.center.y == INT_MAX || t.circumcircle.center.y == INT_MIN))
        {
            continue;
        }
        draw_circle(ren, t.circumcircle.center.x, t.circumcircle.center.y, (int)t.circumcircle.radius);
    }

    // Draw triangles.
    for (int i = 0; i < triangle_count; i += 1)
    {
        Triangle t = triangulation[i];
        if (t.evicted) continue;
        if (t.evicted)
        {
            SDL_SetRenderDrawColor(ren, 255, 0, 0, 255);
        }
        else
        {
            SDL_SetRenderDrawColor(ren, (Uint8)foreground.r, (Uint8)foreground.g, (Uint8)foreground.b, 255);
        }

        Point a, b, c;
        a = t.points[0]; 
        b = t.points[1]; 
        c = t.points[2];

        SDL_RenderDrawLine(ren, a.x, a.y, b.x, b.y);
        SDL_RenderDrawLine(ren, b.x, b.y, c.x, c.y);
        SDL_RenderDrawLine(ren, c.x, c.y, a.x, a.y);
    }

    for (int i = 0; i < point_count; i += 1)
    {
        Point p = points[i];
        SDL_SetRenderDrawColor(ren, (Uint8)p.r, (Uint8)p.g, (Uint8)p.b, 255);
        draw_circle(ren, p.x, p.y, 2);
    }

    SDL_RenderPresent(ren);

    // free(points);
    // free(triangulation);
}

void voronoi_naive(Point *points, char *pixels)
{
    for (int i = 0; i < SCREEN_W; i += 1)
    {
        for (int j = 0; j < SCREEN_H; j += 1)
        {
            float minimum_distance = FLT_MAX;
            int minimum_index = 0;
	
			for (int k = 0; k < POINT_COUNT; k += 1)
			{
				// Get distance from each point
                float x_distance = (float)(points[k].x - i);
                float y_distance = (float)(points[k].y - j);
                float distance = (x_distance * x_distance) + (y_distance * y_distance);

                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    minimum_index = k;
                }
			}

			// Draw pixel
            int offset = (j*4) * SCREEN_W + (i*4);
            pixels[offset + 0] = (char)points[minimum_index].b;
            pixels[offset + 1] = (char)points[minimum_index].g;
            pixels[offset + 2] = (char)points[minimum_index].r;
            pixels[offset + 3] = 255;
        }
    }
}

int compare_points_along_x(const void *a, const void *b)
{
    return ((Point *)a)->x < ((Point *)b)->x;
}

void voronoi_grid(Point *points, char *pixels)
{
    // Define the subdivisions of the grid
    int NUM_COLUMNS = 20;
    int NUM_ROWS    = 20;

    int cell_width  = SCREEN_W / NUM_COLUMNS;
    int cell_height = SCREEN_H / NUM_ROWS; 

	for (int i = 0; i < POINT_COUNT; ++i)
	{
        Point *p = &points[i];
        p->cell_x = p->x / cell_width;
        p->cell_y = p->y / cell_height;
	}

    // Now, sort points along X axis 
    qsort(points, POINT_COUNT, sizeof(Point), compare_points_along_x);

    // Iterate over all the pixels
	for (int i = 0; i < SCREEN_W; ++i)
	{
		for (int j = 0; j < SCREEN_H; ++j)
		{
            float minimum_distance = 100000.0f;
            int minimum_index = 0;

            int this_cell_x = i / cell_width;
            int this_cell_y = j / cell_height;
	
			for (int k = 0; k < POINT_COUNT; ++k)
			{
                Point *p = &points[k];

                // Early exit if we're not in an adjacent grid cell
                if (!(abs(p->cell_x-this_cell_x)<=1 && abs(p->cell_y-this_cell_y)<=1))
                {
                    continue;
                }

				// Get distance from each point
                float x_distance = (float)(p->x - i);
                float y_distance = (float)(p->y - j);
                float distance = (x_distance * x_distance) + (y_distance * y_distance); 

                if (distance < minimum_distance)
                {
                    minimum_distance = distance;
                    minimum_index = k;
                }
			}

			// Draw pixel
            int offset = (j*4) * SCREEN_W + (i*4);
            pixels[offset + 0] = (char)points[minimum_index].b;
            pixels[offset + 1] = (char)points[minimum_index].g;
            pixels[offset + 2] = (char)points[minimum_index].r;
            pixels[offset + 3] = 255;
        }
	}
}

int main(int argc, char *argv[])
{ 
    (void)argc;
    (void)argv;

    // Turn off stdout buffering so printf works.
    setvbuf(stdout, NULL, _IONBF, 0);

	SDL_Init(SDL_INIT_EVERYTHING);

	// Setup window
	SDL_Window *win = SDL_CreateWindow("Voronoi",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
			SCREEN_W, SCREEN_H,
			SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

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
    bool quit = false;
	bool doiter = false;
	int frame = 0;
	char frame_s[10];
    char mode_s[20];
	char point_s[30];
    bool show_text = true;

    int mode = 0;

	int point_count = POINT_COUNT;

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
                                doiter = true;
                                break;

                            case SDLK_TAB:
                                show_text = !show_text;
                                break;

                            case SDLK_0:
                                mode = 0;
                                break;

                            case SDLK_1:
                                mode = 1;
                                break;

                            case SDLK_9:
                                mode = 9;
                                break;

                            case SDLK_ESCAPE:
                                quit = true;
                                break;
                        }
                        break;
                    case SDL_QUIT:
                        quit = true;
                        break;
                    default:
                        break;
                }
            }

            if (doiter && !quit) 
            {
                doiter = false;

                start_time = SDL_GetTicks();

                // Render
                SDL_RenderClear(ren);

                // Create texture
                char *pixels = malloc(4 * SCREEN_W * SCREEN_H);
                SDL_Texture *texture = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_W, SCREEN_H);

                // Assign points
                Point *points = malloc(sizeof(Point) * point_count);
                for (int i = 0; i < point_count; i += 1)
                {
                    points[i].x = rand() % SCREEN_W;
                    points[i].y = rand() % SCREEN_H;
                    points[i].r = rand() % 255;
                    points[i].g = rand() % 255;
                    points[i].b = rand() % 255;
                }
                
                switch (mode)
                {
                    case 1:
                        voronoi_grid(points, pixels);
                        break;
                    case 9:
                        voronoi_cuda(points, pixels);
                    break;
                    case 0:
                    default:
                        voronoi_naive(points, pixels);
                        break;
                }

                // Render
                SDL_RenderClear(ren);
                SDL_UpdateTexture(texture, NULL, pixels, SCREEN_W*4);

                SDL_RenderCopy(ren, texture, NULL, NULL);

                SDL_DestroyTexture(texture);
                free(points);
                free(pixels);
                
                snprintf(frame_s, sizeof(frame_s), "%u ms", frame_time);
                snprintf(mode_s, sizeof(mode_s), "%u mode", mode);
                snprintf(point_s, sizeof(point_s), "%d points", point_count);

                SDL_Surface *frame_surface = TTF_RenderText_Solid(font, frame_s, font_color);
                SDL_Texture *frame_texture = SDL_CreateTextureFromSurface(ren, frame_surface);
                int frame_x, frame_y;
                SDL_QueryTexture(frame_texture, NULL, NULL, &frame_x, &frame_y);
                SDL_Rect frame_rect = {5 , 5, frame_x, frame_y};

                if (show_text) {
                    SDL_RenderCopy(ren, frame_texture, NULL, &frame_rect);
                }

                SDL_Surface *mode_surface = TTF_RenderText_Solid(font, mode_s, font_color);
                SDL_Texture *mode_texture = SDL_CreateTextureFromSurface(ren, mode_surface);
                int mode_x, mode_y;
                SDL_QueryTexture(mode_texture, NULL, NULL, &mode_x, &mode_y);
                SDL_Rect mode_rect = {5 , 5 + 10, mode_x, mode_y};

                if (show_text) {
                    SDL_RenderCopy(ren, mode_texture, NULL, &mode_rect);
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

                SDL_FreeSurface(point_surface);
                SDL_DestroyTexture(point_texture);

                end_time = SDL_GetTicks();
                frame_time = end_time - start_time;
            }
        }
    }

	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}
