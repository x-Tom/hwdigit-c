#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#include "math_functions.h"

#define MAX_THREADS 8
#define RNG_STDDEV 1.0

/// TODO 
// BEST TEST 0.84
// WERE NOT USING SOFTMAX OR CROSS ENTROPY LOSS BUT WE GET 0.84
// CHANGE ALL ACTIVATIONS TO RELU EXCEPT SOFTMAX FOR LAST LAYER
// CHANGE COST FUNCTION TO CROSS ENTROPY LOSS

typedef float(*activation_func)(float);

struct neuron_layer {
    struct neuron_layer* next; 
    struct neuron_layer* prev;
    activation_func f;
    activation_func df;
    int na; // number of neurons
    float* a; // f(z), activations (neuron values)
    float* da; // f'(z)
    float* b; // biases
    int nw; // number of elements in weights matrix
    float* w; // weights array (matrix)
    int wdim[2]; // wdim[0]: na, wdim[1]: na of previous layer
    // todo: removed wdim add int na_prev
};

// struct nnbpbpt_args {
//     struct neuron_layer* output_layer;
//     struct neuron_layer* input_layer;
//     int l;
//     int nlayers;
//     int batch_size;
//     int batch_start;
//     float** samples;
//     float** labels;
//     float* success;
// };

enum {
    MEAN_SQUARE_ERROR,
    CROSS_ENTROPY_SOFTMAX_LOSS
};

struct neuron_layer* linked_list_tail(struct neuron_layer* nl);

struct neuron_layer* alloc_neuron_layer(int adata_length, float* adata, int wdata_length, float* wdata, float* bdata, float* ddata, struct neuron_layer* next, struct neuron_layer* prev, activation_func func, activation_func deriv);
void print_neuron_layer_info(struct neuron_layer* nl, int i);
struct neuron_layer* alloc_neural_network(int nlayers, int* dims, float* input_data, activation_func* fs, activation_func* dfs);
void print_neural_network_info(struct neuron_layer* input_layer);
struct neuron_layer* neural_network_forward_pass(struct neuron_layer* input_layer);
struct neuron_layer* neural_network_forward_pass_inputs(struct neuron_layer* input_layer, float* inputs);

cost_act_grad switch_cost(uint cost_function);

struct neuron_layer* neural_network_back_propagation_stochastic(struct neuron_layer* output_layer, float* y, int l, int nlayers, uint cost_function, float learning_rate);
float neural_network_back_propagation_batch(struct neuron_layer* output_layer, struct neuron_layer* input_layer, /*float* y,*/ int l, int nlayers, int batch_size, float** samples, float** labels, uint cost_function, float learning_rate);

struct neuron_layer* neural_network_back_propagation_batch_iterative(struct neuron_layer* output_layer, struct neuron_layer* input_layer, /*float* y,*/ int nlayers, int batch_size, float** samples, float** labels, uint cost_function, float learning_rate);

struct neuron_layer* neural_network_back_propagation_stochastic_iterative(struct neuron_layer* output_layer, struct neuron_layer* input_layer, int nlayers, int set_size, float** samples, float** labels, float* success, uint cost_function, float learning_rate);


// void neural_network_back_propagation_batch_pthreads(void* args);
// void neural_network_back_propagation_stochastic_pthreads(void* args);

// void neural_network_train_batch_pthreads(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int batch_size, int set_size, int epochs, float** samples, float** labels);
// float neural_network_train_stochastic_pthreads(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int set_size, int epochs, float** samples, float** labels);

float neural_network_predict(struct neuron_layer* input_layer, float* sample, float* label);


void neural_network_train_batch(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int batch_size, int set_size, int epochs, float** samples, float** labels, uint cost_function, float learning_rate);
void neural_network_train_stochastic(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int set_size, int epochs, float** samples, float** labels, uint cost_functions, float learning_rate);

float neural_network_test(struct neuron_layer* input_layer, struct neuron_layer* output_layer, float** samples, float** labels, int test_size);

char* neural_network_info_string(char* str, int* dims, int n);