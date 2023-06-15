#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdlib.h>
#include <time.h>
#include "aux.h"

#define BUF_SIZE 128

#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define SCALE_FACTOR 10
#define WINDOW_WIDTH 560
#define WINDOW_HEIGHT 280
#define TEXT_POS_X 300
#define TEXT_POS_Y 70

// SDL_Color textColor = { 255, 255, 255, 0 }; // white

void normalise_mnist_pixel_data(float** arr, int n);

float** read_mnist_label_data(const char* file_name, int num_samples);
// uint8_t*** read_mnist_pixel_data(const char* file_name, int num_samples, float*** images);
float** read_mnist_pixel_data(const char* file_name, int num_samples);
SDL_Texture* createGrayscaleImageTexture(float* array, SDL_Renderer *renderer);
// void print_mnist_data(float* array, int amt);
// void print_mnist_label_data(float* array, int amt);
void print_mnist_data(float** array, int amt);
void print_mnist_label_data(float** array, int amt);
void print_mnist_label_data_raw(float** array, int amt);