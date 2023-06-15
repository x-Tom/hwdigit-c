#include "neural_net.h"
#include "display.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TRAIN_SET_SIZE 60000
#define TEST_SET_SIZE 10000

#define LEARNING_RATE (float)0.00005
#define EPOCHS 200
// #define BATCH_SIZE 32

#define FONT_SIZE 22
#define FONT_SIZE_SMALL 12

SDL_Color textColor = { 255, 255, 255, 0 }; // white

int main(int argc, char** argv) {

    srand(time(NULL));

    float** x_train = read_mnist_pixel_data("x_train.txt", 60000);
    float** y_train = read_mnist_label_data("y_train.txt", 60000);

    float** x_test = read_mnist_pixel_data("x_test.txt", 10000);
    float** y_test = read_mnist_label_data("y_test.txt", 10000);

    int layer_dims[] = {IMG_HEIGHT*IMG_WIDTH,16,16,10};
    int nlayers = sizeof(layer_dims)/sizeof(*layer_dims);
    activation_func activation_functions[] = {NULL, relu, relu, expf};
    activation_func activation_derivatives[] = {NULL, step, step, unity};

    float inputs[IMG_HEIGHT*IMG_WIDTH] = {0}; 


    
    struct neuron_layer* nnet = alloc_neural_network(4, layer_dims, inputs, activation_functions, activation_derivatives);
    struct neuron_layer* nnet_output = linked_list_tail(nnet);

    neural_network_train_stochastic(nnet, nnet_output, nlayers, TRAIN_SET_SIZE, EPOCHS, x_train, y_train, MEAN_SQUARE_ERROR, LEARNING_RATE);

    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;
    SDL_Texture *texture = NULL;
    TTF_Font *font = NULL;
    TTF_Font *font_small = NULL;
    
    SDL_Color textColor = { 255, 255, 255, 0 }; // white
    SDL_Color textGreen = {   0, 255,   0, 0 };
    SDL_Color textRed =   { 255,   0,   0, 0 };

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Could not initialize sdl2: %s\n", SDL_GetError());
        return 1;
    }

    if (TTF_Init() == -1) {
        printf("TTF_Init: %s\n", TTF_GetError());
        return 2;
    }

    window = SDL_CreateWindow("Handwritten Digit Classifier", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL) {
        printf("Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL) {
        printf("Could not create renderer: %s\n", SDL_GetError());
        return 1;
    }
    
    font = TTF_OpenFont("sf-pro-text-regular.ttf", FONT_SIZE);
    if (font == NULL) {
        printf("Could not load font: %s\n", TTF_GetError());
        return 1;
    }

    font_small = TTF_OpenFont("sf-pro-text-regular.ttf", FONT_SIZE_SMALL);
    if (font == NULL) {
        printf("Could not load font: %s\n", TTF_GetError());
        return 1;
    }

    
    char infoString[BUF_SIZE] = {0};
    neural_network_info_string(infoString, layer_dims, sizeof(layer_dims)/sizeof(*layer_dims));
    printf("%s", infoString);


    SDL_Rect srcRect = { 0, 0, IMG_WIDTH, IMG_HEIGHT };
    SDL_Rect destRect = { 0, 0, IMG_WIDTH * SCALE_FACTOR, IMG_HEIGHT * SCALE_FACTOR };

    SDL_Event event;
    int running = 1;
    int counter = 0;
    char str[BUF_SIZE];
    char str2[BUF_SIZE];
    char str3[BUF_SIZE];
    float correct = 0;
    float accuracy = 0;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            }
        }

        SDL_Surface *text_surface = NULL;
        SDL_Texture *text_texture = NULL;

        SDL_Surface *text_surface2 = NULL;
        SDL_Texture *text_texture2 = NULL;
        SDL_Color textcol2;

        SDL_Surface *text_surface3 = NULL;
        SDL_Texture *text_texture3 = NULL;

        SDL_Surface *text_surface4 = NULL;
        SDL_Texture *text_texture4 = NULL;


        if ((uint)neural_network_predict(nnet, x_test[counter], y_test[counter])) {
            correct++;
            textcol2 = textGreen;
        } else textcol2 = textRed;

        accuracy = correct/(float)(counter+1);

        int predicted = max_array_index(nnet_output->a, nnet_output->na);
        int actual = max_array_index(y_test[counter], 10);

        sprintf(str, "Label: %hhu", (uint8_t)actual);
        sprintf(str2, "Predicted: %hhu", (uint8_t)predicted);
        sprintf(str3, "Accuracy: %f\n%d/%d", accuracy, (int)correct, counter+1);


        text_surface = TTF_RenderText_Blended(font, str, textColor);
        if (text_surface == NULL) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
        if (text_texture == NULL) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }

        text_surface2 = TTF_RenderText_Blended(font, str2, textcol2);
        if (text_surface2 == NULL) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture2 = SDL_CreateTextureFromSurface(renderer, text_surface2);
        if (text_texture2 == NULL) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }

        text_surface3 = TTF_RenderText_Blended_Wrapped(font, str3, textColor, 0);
        if (text_surface3 == NULL) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture3 = SDL_CreateTextureFromSurface(renderer, text_surface3);
        if (text_texture3 == NULL) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }

        text_surface4 = TTF_RenderText_Blended_Wrapped(font_small, infoString, textColor, 0);
        if (text_surface4 == NULL) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture4 = SDL_CreateTextureFromSurface(renderer, text_surface4);
        if (text_texture4 == NULL) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }


        texture = createGrayscaleImageTexture(x_test[counter], renderer);
        if (texture == NULL) {
            return 1;
        }

        SDL_FreeSurface(text_surface);
        SDL_Rect textRect = { TEXT_POS_X, TEXT_POS_Y, text_surface->w, text_surface->h };
        SDL_Rect textRect2 = { TEXT_POS_X, TEXT_POS_Y+40, text_surface2->w, text_surface2->h };
        SDL_Rect textRect3 = { TEXT_POS_X, TEXT_POS_Y+120, text_surface3->w, text_surface3->h };
        SDL_Rect textRect4 = { TEXT_POS_X, TEXT_POS_Y-65, text_surface4->w, text_surface4->h };



        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_RenderCopy(renderer, texture, &srcRect, &destRect);
        SDL_RenderCopy(renderer, text_texture, NULL, &textRect);
        SDL_RenderCopy(renderer, text_texture2, NULL, &textRect2);
        SDL_RenderCopy(renderer, text_texture3, NULL, &textRect3);
        SDL_RenderCopy(renderer, text_texture4, NULL, &textRect4);
        

        SDL_RenderPresent(renderer);

        if(++counter > TRAIN_SET_SIZE) break;
        usleep(100000);
    }

    TTF_CloseFont(font);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();



    return 0;
}

