#include "display.h"
// #include "aux.h"


float** read_mnist_label_data(const char* file_name, int num_samples){
    FILE* file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Could not open file\n");
        return NULL;
    }
    float** labels = (float**)malloc(sizeof(float*)*num_samples);
    
    for (int i = 0; i < num_samples; i++) { 
        Uint8 digit = 0;
        if (!fscanf(file, "%hhu", &digit)) {
            printf("Error reading file\n");
            return NULL;
        }
        labels[i] = (float*)calloc(10,sizeof(float));
        labels[i][(int)digit] = 1;
    }
    return labels;
}



float** read_mnist_pixel_data(const char* file_name, int num_samples){
    FILE* file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Could not open file\n");
        return NULL;
    }

    // float*[IMG_HEIGHT][IMG_WIDTH]
    float** imgs = (float**)malloc(num_samples*sizeof(float*));
    for (int i = 0; i < num_samples; i++) {
        *(imgs+i) = (float*)malloc(IMG_HEIGHT*IMG_WIDTH*sizeof(float));
    }



    // float images[num_samples][IMG_HEIGHT][IMG_WIDTH]; // Assuming we have 10 images
    for (int k = 0; k < num_samples; k++) { // for each image
        for (int i = 0; i < IMG_HEIGHT; i++) { // for each row
            for (int j = 0; j < IMG_WIDTH; j++) { // for each column
                if (!fscanf(file, "%f", &imgs[k][i*IMG_WIDTH + j])) {
                    printf("Error reading file\n");
                    return NULL;
                }
                imgs[k][i*IMG_WIDTH + j] /= (float)255;
            }
        }
    }

    fclose(file);

    return imgs;
}
// void normalise_mnist_pixel_data(float** arr, int n){
//     for(int i = 0; i < n; i++){
//         for(int j = 0; j < IMG_HEIGHT*IMG_WIDTH; j++){
//             arr[i][j] = (float)(arr[i][j]/255);
//         }
//     }
// }



SDL_Texture* createGrayscaleImageTexture(float* array, SDL_Renderer *renderer) {
    // Initialize random number generator
    srand(time(NULL));

    // Create a surface to hold the grayscale image
    SDL_Surface* surface = SDL_CreateRGBSurface(0, IMG_WIDTH, IMG_HEIGHT, 32, 0, 0, 0, 0);
    if (surface == NULL) {
        printf("Could not create surface: %s\n", SDL_GetError());
        return NULL;
    }

    // Fill the surface with random grayscale values
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            Uint8 gray = (Uint8)(255*array[y * IMG_WIDTH + x]); // Random gray value
            Uint32 color = SDL_MapRGB(surface->format, gray, gray, gray);
            ((Uint32*)surface->pixels)[y * IMG_WIDTH + x] = color;
        }
    }

    // Create a texture from the surface
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (texture == NULL) {
        printf("Could not create texture: %s\n", SDL_GetError());
        return NULL;
    }

    // We don't need the surface anymore
    SDL_FreeSurface(surface);

    // Return the texture
    return texture;
}

// void print_mnist_data(float* array, int amt){
//     for (int i = 0; i < amt; i++) {
//         for (int y = 0; y < IMG_HEIGHT; y++) {
//             for (int x = 0; x < IMG_WIDTH; x++) {
//                 Uint8 gray = (Uint8)array[i*IMG_HEIGHT*IMG_WIDTH + y*IMG_WIDTH+x]; // Random gray value
//                 printf("%hhu ", gray);
//             }
//             printf("\n");
//         }
//         printf("\n\n--------------------------------------\n\n");
//     }
// }

void print_mnist_data(float** array, int amt){
    for (int i = 0; i < amt; i++) {
        for (int y = 0; y < IMG_HEIGHT; y++) {
            for (int x = 0; x < IMG_WIDTH; x++) {
                // Uint8 gray = (Uint8)array[i][y*IMG_WIDTH+x]; // Random gray value
                // printf("%hhu ", gray);
                printf("%f ", array[i][y*IMG_WIDTH+x]);
            }
            printf("\n");
        }
        printf("\n\n--------------------------------------\n\n");
    }
}

// void print_mnist_label_data(float* array, int amt){
//     for (int i = 0; i < amt; i++) {
//         printf("%hhu\n", (Uint8)max_array_index(array+i*10, 10));
//     }
// }

void print_mnist_label_data(float** array, int amt){
    for (int i = 0; i < amt; i++) {
        printf("%hhu\n", (Uint8)max_array_index(array[i], 10));
    }
}

void print_mnist_label_data_raw(float** array, int amt){
    for (int i = 0; i < amt; i++) {
        for (int j = 0; j < 10; j++){
            printf("%hhu, ", (Uint8)array[i][j]);   
        }
        printf("\n");
    }
}