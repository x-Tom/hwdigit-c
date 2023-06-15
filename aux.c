#include "aux.h"

void print_arr(float* arr, int n){
    for(int i = 0; i<n; i++){
        printf("%f\n", arr[i]);
    }
}


int max_array_index(float* array, int n){
    float max = array[0];
    int idx = 0;
    for(int i = 1; i < n; i++){
        if(array[i]>max) {
            max = array[i];
            idx = i;
        }
    }
    // if(!max) return -1;
    return idx;
}

void two_array_shuffle(float **arr1, float **arr2, int size){
    if(size <= 0) {
        printf("Error: Array sizes are not valid!\n");
        exit(1);
    }
    srand(time(NULL));

    for(int i = 0; i < size; i++){
        int index = rand() % (size - i) + i;

        float *temp1 = arr1[i];
        arr1[i] = arr1[index];
        arr1[index] = temp1;

        float *temp2 = arr2[i];
        arr2[i] = arr2[index];
        arr2[index] = temp2;
    }
}