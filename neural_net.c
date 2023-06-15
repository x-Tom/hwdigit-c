#include "neural_net.h"
#include "aux.h"

// struct neuron_layer {
//     struct neuron_layer* next; 
//     struct neuron_layer* prev;
//     activation_func f;
//     activation_func df;
//     int na; // number of neurons
//     float* a; // f(z), activations (neuron values)
//     float* da; // f'(z)
//     float* b; // biases
//     int nw; // number of elements in weights matrix
//     float* w; // weights array (matrix)
//     int wdim[2]; // wdim[0]: na, wdim[1]: na of previous layer
// };


struct neuron_layer* alloc_neuron_layer(int adata_length, float* adata, int wdata_length, float* wdata, float* bdata, float* ddata, struct neuron_layer* next, struct neuron_layer* prev, activation_func func, activation_func deriv){
    struct neuron_layer* nl = (struct neuron_layer*)malloc(sizeof(struct neuron_layer));
    nl->next = next;
    nl->prev = prev;
    nl->na = adata_length;
    nl->f = func;
    nl->df = deriv;
    nl->a = (float*)malloc(nl->na*sizeof(float));
    nl->b = (bdata) ? (float*)malloc(nl->na*sizeof(float)) : NULL;
    nl->da = (ddata) ? (float*)malloc(nl->na*sizeof(float)) : NULL;
    nl->nw = wdata_length;
    nl->w = (float*)malloc(nl->nw*sizeof(float));
    memcpy(nl->a, adata, nl->na*sizeof(float));
    memcpy(nl->b, bdata, nl->na*sizeof(float));
    memcpy(nl->da, ddata, nl->na*sizeof(float));
    memcpy(nl->w, wdata, nl->nw*sizeof(float));
    return nl;
}

struct neuron_layer* linked_list_tail(struct neuron_layer* nl){
    struct neuron_layer* tl = nl;
    for(;tl->next!=NULL; tl=tl->next);
    return tl;
}

void print_net_activations(struct neuron_layer* input, int show_inputs){
    int l = 0;
    struct neuron_layer* nl;
    for(nl = (show_inputs) ? input : input->next; nl != NULL; nl = nl->next, l++){
        // printf("Neuron Layer (%d):\nLayer Size: %d\nnext: %p\nprev: %p\nweights (%dx%d):\n", l, nl->na, nl->next, nl->prev, nl->wdim[0], nl->wdim[1]);
        printf("Neuron Layer (%d):\n", l);
        printf("Neuron activations:\n");
        for(int i = 0; i < nl->na; i++) {
            printf("a%d: %f\n",i,nl->a[i]);
        }
    }
}

void print_neuron_layer_info(struct neuron_layer* nl, int i){
    printf("Neuron Layer (%d):\nLayer Size: %d\nnext: %p\nprev: %p\nweights (%dx%d):\n", i, nl->na, nl->next, nl->prev, nl->wdim[0], nl->wdim[1]);
    int d0 = nl->wdim[0];
    int d1 = nl->wdim[1];
    for(int i = 0; i < d0; i++) {
        for (int j = 0; j < d1; j++){
            printf("w%d,%d: %f\n",i,j,nl->w[j+i*d1]);
        }
    }
    printf("biases:\n");
    for(int i = 0; i < nl->na; i++) {
        if(nl->b!=NULL) printf("b%d: %f\n",i,nl->b[i]);
    }
    printf("Neuron activations:\n");
    for(int i = 0; i < nl->na; i++) {
        printf("a%d: %f\n",i,nl->a[i]);
    }
}

struct neuron_layer* alloc_neural_network(int nlayers, int* dims, float* input_data, activation_func* fs, activation_func* dfs){

    struct neuron_layer* input_layer = alloc_neuron_layer(dims[0], input_data, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    input_layer->wdim[0] = 0;
    input_layer->wdim[1] = 0;

    struct neuron_layer* crnt_layer = input_layer;

    for(int i = 1; i < nlayers; i++){

        
        int nw = dims[i]*dims[i-1];
        float* w = rand_gauss_weights(nw, 0, RNG_STDDEV);
        float* b = rand_gauss_weights(dims[i], 0, RNG_STDDEV);
        float* a = (float*)calloc(dims[i], sizeof(float));
        float* da = (float*)calloc(dims[i], sizeof(float));

        crnt_layer->next = alloc_neuron_layer(dims[i], a, nw, w, b, da, NULL, crnt_layer, fs[i], dfs[i]);

        crnt_layer->next->wdim[0] = dims[i];
        crnt_layer->next->wdim[1] = dims[i-1];
        
        crnt_layer = crnt_layer->next;

        free(w);
        free(a);
        free(b);
        free(da);
    }

    return input_layer;
}



void print_neural_network_info(struct neuron_layer* input_layer){
    struct neuron_layer* crnt_layer = input_layer; 
    for(int i = 0; crnt_layer!=NULL; crnt_layer = crnt_layer->next, i++){
        print_neuron_layer_info(crnt_layer,i);
    }
}

struct neuron_layer* neural_network_forward_pass(struct neuron_layer* input_layer){
    struct neuron_layer* crnt_layer = NULL;
    struct neuron_layer* layer = NULL;
    for(crnt_layer = input_layer->next; crnt_layer!=NULL; crnt_layer = crnt_layer->next) {
        // Wij row major order
        int dim0 = crnt_layer->wdim[0];
        int dim1 = crnt_layer->wdim[1];
        for(int i = 0; i < dim0; i++) {
            // z
            crnt_layer->a[i] = dot(crnt_layer->w + i*dim1, crnt_layer->prev->a, dim1) + crnt_layer->b[i];
            crnt_layer->da[i] = crnt_layer->a[i];
            //ai = σ(wi.ai_prev + bi);
        }
        // apply activation function and cache derivative
        if(crnt_layer->next == NULL && (crnt_layer->f == expf && crnt_layer->df == unity)) { // last layer and softmax case
            element_wise_mut_val_sub(max_coeff_arr(crnt_layer->a,crnt_layer->na), crnt_layer->a, crnt_layer->na);
            element_wise_mut_func(crnt_layer->f, crnt_layer->a, crnt_layer->na);
            softmax(crnt_layer->a, crnt_layer->na);
        } else { // usual case
            element_wise_mut_func(crnt_layer->f, crnt_layer->a, crnt_layer->na);
        }
        element_wise_mut_func(crnt_layer->df, crnt_layer->da, crnt_layer->na);
        layer = crnt_layer;
    }
    return layer;
}

struct neuron_layer* neural_network_forward_pass_inputs(struct neuron_layer* input_layer, float* inputs){
    for (int i = 0; i < input_layer->na; i++) {
        input_layer->a[i] = inputs[i];
    }
    return neural_network_forward_pass(input_layer);
}

cost_act_grad switch_cost(uint cost_function){
    cost_act_grad cost_activation_gradient = NULL;
    switch(cost_function){
        case CROSS_ENTROPY_SOFTMAX_LOSS:
            cost_activation_gradient = cost_activation_gradient_mse;
            break;
        default:
        case MEAN_SQUARE_ERROR:
            cost_activation_gradient = cost_activation_gradient_mse;
            break;
    }
    return cost_activation_gradient;
}


struct neuron_layer* neural_network_back_propagation_stochastic(struct neuron_layer* output_layer, float* y, int l, int nlayers, uint cost_function, float learning_rate){
    cost_act_grad cost_activation_gradient = switch_cost(cost_function);
    struct neuron_layer* crnt_layer = output_layer;
    // ∇_aC transposed?
    // δ = ∇_aC ⊙ σ'(z)
    static float* delta = NULL;

    // printf("passed\n");


    int dim0 = crnt_layer->wdim[0];
    int dim1 = crnt_layer->wdim[1];

    float nu = learning_rate;

    if(l == nlayers-1) {
        // printf("passed\n");
        delta = cost_activation_gradient(crnt_layer->a,y,(float*)malloc(crnt_layer->na*sizeof(float)), crnt_layer->na);
        // printf("passed\n");
        delta = hadamard_product(delta,crnt_layer->da, crnt_layer->na);
        // printf("passed\n");
    } else if (l == 0){
        //...
        free(delta);
        delta = NULL;
        return crnt_layer;
    } else {
        // δ^l = ((W^(l+1)^T)δ^(l+1)) ⊙ σ'(z)
        int ndim0 = crnt_layer->next->wdim[0];
        int ndim1 = crnt_layer->next->wdim[1];
        float* old_delta = delta;
        // printf("passedl-1\n");

        delta = matrix_multiply_transpose(crnt_layer->next->w, ndim0, ndim1, delta, ndim0, 1, (float*)malloc(crnt_layer->na*sizeof(float)));
        delta = hadamard_product(delta, crnt_layer->da, crnt_layer->na);
        free(old_delta);
    }

    for(int i = 0; i < dim0; i++){
        for (int j = 0; j < dim1; j++){
            // printf("error?\n");
            crnt_layer->w[j + i*dim1] -= nu*delta[i]*crnt_layer->prev->a[j];
        }
        crnt_layer->b[i] -= nu*delta[i];
    }

    // printf("recursion\n");


    return neural_network_back_propagation_stochastic(crnt_layer->prev, y, l-1, nlayers, cost_function, learning_rate);
 
}

struct neuron_layer* neural_network_back_propagation_stochastic_iterative(struct neuron_layer* output_layer, struct neuron_layer* input_layer, int nlayers, int set_size, float** samples, float** labels, float* success, uint cost_function,float learning_rate){
    cost_act_grad cost_activation_gradient = switch_cost(cost_function);

    // ∇_aC transposed?
    // δ = ∇_aC ⊙ σ'(z)

    float nu = learning_rate;

    float correct = 0;
    
    for(int b = 0; b < set_size; b++){

        float* delta = NULL;
        struct neuron_layer* crnt_layer = output_layer;

        neural_network_forward_pass_inputs(input_layer, samples[b]);
        if(max_array_index(crnt_layer->a, 10) == max_array_index(labels[b], 10)) correct++;



        for(int l = nlayers-1; l >= 0; l--, crnt_layer = crnt_layer->prev){

            int dim0 = crnt_layer->wdim[0];
            int dim1 = crnt_layer->wdim[1];

            if(l == nlayers-1) {
                // printf("passed\n");
                delta = cost_activation_gradient(crnt_layer->a,labels[b],(float*)malloc(crnt_layer->na*sizeof(float)), crnt_layer->na);
                // printf("passed\n");
                delta = hadamard_product(delta,crnt_layer->da, crnt_layer->na);
                // printf("passed\n");
            } else if (l == 0){
                //...
                free(delta);
                delta = NULL;
                if(success!=NULL) *success = correct / (float)set_size;
                return crnt_layer;
            } else {
                // δ^l = ((W^(l+1)^T)δ^(l+1)) ⊙ σ'(z)
                int ndim0 = crnt_layer->next->wdim[0];
                int ndim1 = crnt_layer->next->wdim[1];
                float* old_delta = delta;
                // printf("passedl-1\n");

                delta = matrix_multiply_transpose(crnt_layer->next->w, ndim0, ndim1, delta, ndim0, 1, (float*)malloc(crnt_layer->na*sizeof(float)));
                delta = hadamard_product(delta, crnt_layer->da, crnt_layer->na);
                free(old_delta);
            }

            for(int i = 0; i < dim0; i++){
                for (int j = 0; j < dim1; j++){
                    // printf("error?\n");
                    crnt_layer->w[j + i*dim1] -= nu*delta[i]*crnt_layer->prev->a[j];
                }
                crnt_layer->b[i] -= nu*delta[i];
            }
        }
    }

    // printf("recursion\n");


    // return crnt_layer;
 
}

float neural_network_back_propagation_batch(struct neuron_layer* output_layer, struct neuron_layer* input_layer, /*float* y,*/ int l, int nlayers, int batch_size, float** samples, float** labels, uint cost_function, float learning_rate){
    cost_act_grad cost_activation_gradient = switch_cost(cost_function);

    static int correct = 0;
    
    struct neuron_layer* crnt_layer = output_layer;
    // ∇_aC transposed?
    // δ = ∇_aC ⊙ σ'(z)
    static float** delta = NULL; // delta per batch at current layer
    static float** cost_weight_jacobian = NULL; // cost weight jacobian per layer, used to sum sample cost weight jacobians
    static float** cost_bias_jacobian = NULL; 

    // printf("passed\n");

    int dim0 = crnt_layer->wdim[0];
    int dim1 = crnt_layer->wdim[1];

    float nu = learning_rate;

    if(l == nlayers-1) {

        // sample_batch = flatten_image_samples(samples, batch_size);        
        cost_weight_jacobian = (float**)malloc(l*sizeof(float*) - 1);
        cost_bias_jacobian = (float**)malloc(l*sizeof(float*) - 1);
        struct neuron_layer* nl = crnt_layer;
        for (int i = nlayers-1; i > 0; i--, nl = nl->prev){
            cost_weight_jacobian[i-1] = (float*)calloc(nl->nw,sizeof(float));
            cost_bias_jacobian[i-1] = (float*)calloc(nl->na,sizeof(float));
        }
        
        delta = malloc(batch_size*sizeof(float*));

    }

    for(int b = 0; b < batch_size; b++){

        // neural_network_forward_pass_inputs(input_layer, samples[b]);
        correct += neural_network_predict(input_layer, samples[b], labels[b]);


        if(l == nlayers-1) {
            // printf("passed\n");
            delta[b] = cost_activation_gradient(crnt_layer->a,labels[b],(float*)malloc(crnt_layer->na*sizeof(float)), crnt_layer->na);
            // printf("passed\n");
            delta[b] = hadamard_product(delta[b],crnt_layer->da, crnt_layer->na);
            // printf("passed\n");
        } else if (l == 0){
            free(delta[b]);
            delta[b] = NULL;
        } else {
            // δ^l = ((W^(l+1)^T)δ^(l+1)) ⊙ σ'(z)
            int ndim0 = crnt_layer->next->wdim[0];
            int ndim1 = crnt_layer->next->wdim[1];
            float* old_delta = delta[b];
            // printf("passedl-1\n");

            delta[b] = matrix_multiply_transpose(crnt_layer->next->w, ndim0, ndim1, delta[b], ndim0, 1, (float*)malloc(crnt_layer->na*sizeof(float)));
            delta[b] = hadamard_product(delta[b], crnt_layer->da, crnt_layer->na);
            free(old_delta);
        }
    
        for(int i = 0; i < dim0; i++){
            for (int j = 0; j < dim1; j++){
                cost_weight_jacobian[l-1][j + i*dim1] += delta[b][i]*crnt_layer->prev->a[j];
            }
            cost_bias_jacobian[l-1][i] += delta[b][i];            
        }
    }


    if(l == 0){

        for(int i = nlayers-1; i > 0; i--){
            free(cost_weight_jacobian[i-1]);
            free(cost_bias_jacobian[i-1]);
        }
        free(cost_weight_jacobian);
        cost_weight_jacobian = NULL;
        free(cost_bias_jacobian);
        cost_bias_jacobian = NULL;
        float crct = (float)correct;
        correct = 0;
        return crct;
    }

    for(int i = 0; i < dim0; i++){
        for (int j = 0; j < dim1; j++){
            // printf("error?\n");
            crnt_layer->w[j + i*dim1] -= (1/batch_size)*nu*cost_weight_jacobian[l-1][j + i*dim1];
        }
        crnt_layer->b[i] -= (1/batch_size)*nu*cost_bias_jacobian[l-1][i];
    }

    return neural_network_back_propagation_batch(crnt_layer->prev, input_layer, l-1, nlayers, batch_size, samples, labels, cost_function, learning_rate);
 
}

struct neuron_layer* neural_network_back_propagation_batch_iterative(struct neuron_layer* output_layer, struct neuron_layer* input_layer, int nlayers, int batch_size, float** samples, float** labels, uint cost_function, float learning_rate){
    cost_act_grad cost_activation_gradient = switch_cost(cost_function);
    struct neuron_layer* crnt_layer = output_layer;
    // ∇_aC transposed?
    // δ = ∇_aC ⊙ σ'(z)
    float** delta = (float**)malloc(batch_size*sizeof(float*)); // delta per batch at current layer
    // float** cost_weight_jacobian = (float*)malloc(nlayers*sizeof(float*)-1); // cost weight jacobian per layer, used to sum sample cost weight jacobians
    // float** cost_bias_jacobian = (float*)malloc(nlayers*sizeof(float*)-1); 

    // printf("passed\n");

    


    float nu = learning_rate;

    for (int l = nlayers-1; l >= 0; l--, crnt_layer = crnt_layer->prev){
        float* cost_weight_jacobian = (float*)calloc(crnt_layer->nw,sizeof(float));
        float* cost_bias_jacobian = (float*)calloc(crnt_layer->na,sizeof(float));

        int dim0 = crnt_layer->wdim[0];
        int dim1 = crnt_layer->wdim[1];
        
        for(int b = 0; b < batch_size; b++){

            if (l == 0){
                free(delta[b]);
                delta[b] = NULL;
                goto end;
            }

            neural_network_forward_pass_inputs(input_layer, samples[b]);

            if(l == nlayers-1) {
                // printf("passed\n");
                delta[b] = cost_activation_gradient(crnt_layer->a,labels[b],(float*)malloc(crnt_layer->na*sizeof(float)), crnt_layer->na);
                // printf("passed\n");
                delta[b] = hadamard_product(delta[b],crnt_layer->da, crnt_layer->na);
                // printf("passed\n");
            } else {
                // δ^l = ((W^(l+1)^T)δ^(l+1)) ⊙ σ'(z)
                int ndim0 = crnt_layer->next->wdim[0];
                int ndim1 = crnt_layer->next->wdim[1];
                float* old_delta = delta[b];
                // printf("passedl-1\n");

                delta[b] = matrix_multiply_transpose(crnt_layer->next->w, ndim0, ndim1, delta[b], ndim0, 1, (float*)malloc(crnt_layer->na*sizeof(float)));
                delta[b] = hadamard_product(delta[b], crnt_layer->da, crnt_layer->na);
                free(old_delta);
            }
        
            for(int i = 0; i < dim0; i++){
                for (int j = 0; j < dim1; j++){
                    cost_weight_jacobian[j + i*dim1] += delta[b][i]*crnt_layer->prev->a[j];
                }
                cost_bias_jacobian[i] += delta[b][i];            
            }
        }

        for(int i = 0; i < dim0; i++){
            for (int j = 0; j < dim1; j++){
                // printf("error?\n");
                crnt_layer->w[j + i*dim1] -= (1/batch_size)*nu*cost_weight_jacobian[j + i*dim1];
            }
            crnt_layer->b[i] -= (1/batch_size)*nu*cost_bias_jacobian[i];
        }

        end:

        free(cost_weight_jacobian);
        cost_weight_jacobian = NULL;
        free(cost_bias_jacobian);
        cost_bias_jacobian = NULL;

    }
        

    

    return crnt_layer;
}



// struct nnbpbpt_args {
//     struct neuron_layer* output_layer;
//     struct neuron_layer* input_layer;
//     int l;
//     int nlayers;
//     int batch_size;
//     int batch_start;
//     float** samples;
//     float** labels;
// };


// void neural_network_back_propagation_batch_pthreads(void* args){
//     struct nnbpbpt_args* pargs = (struct nnbpbpt_args*)args;
//     neural_network_back_propagation_batch(pargs->output_layer, pargs->input_layer, pargs->l, pargs->nlayers, pargs->batch_size, pargs->samples, pargs->labels);
//     pthread_exit(NULL);
// }

// void neural_network_back_propagation_batch_pthreads(void* args){
//     struct nnbpbpt_args* pargs = (struct nnbpbpt_args*)args;
//     neural_network_back_propagation_batch_iterative(pargs->output_layer, pargs->input_layer, pargs->nlayers, pargs->batch_size, pargs->samples, pargs->labels);
//     pthread_exit(NULL);
// }

// void neural_network_back_propagation_stochastic_pthreads(void* args){
//     struct nnbpbpt_args* pargs = (struct nnbpbpt_args*)args;
//     neural_network_back_propagation_stochastic_iterative(pargs->output_layer, pargs->input_layer, pargs->nlayers, pargs->batch_size, pargs->samples, pargs->labels, pargs->success);
//     pthread_exit(NULL);
// }

// float neural_network_train_stochastic_pthreads(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int set_size, int epochs, float** samples, float** labels){
//     pthread_t threads[MAX_THREADS];
//     struct nnbpbpt_args thread_args[MAX_THREADS] = {0};
//     int chunk_size = set_size / MAX_THREADS;
//     float success = 0;
//     for(int e = 0; e < epochs; e++){
//         for (int i = 0; i < MAX_THREADS; i++) {
//             thread_args[i].input_layer = input_layer;
//             thread_args[i].output_layer = output_layer;
//             thread_args[i].samples = samples;
//             thread_args[i].labels = labels;
//             thread_args[i].l = nlayers-1;
//             thread_args[i].nlayers = nlayers;
//             thread_args[i].batch_size = chunk_size;
//             thread_args[i].batch_start = i * chunk_size;
//             thread_args[i].success = malloc(sizeof(float));

//             int ret = pthread_create(&threads[i], NULL, (void*)neural_network_back_propagation_stochastic_pthreads, (void*)&thread_args[i]);

//             if (ret) {
//                 printf("Error:unable to create thread, %d\n", ret);
//                 exit(-1);
//             }
//         }

//         for(int i = 0; i < MAX_THREADS; i++) {
//             success += *thread_args[i].success;
//             free(thread_args[i].success);
//             pthread_join(threads[i], NULL);
//         }
//     }
//     return success / (float)epochs*MAX_THREADS;
// }

// void neural_network_train_batch_pthreads(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int batch_size, int set_size, int epochs, float** samples, float** labels){
//     pthread_t threads[MAX_THREADS];
//     struct nnbpbpt_args thread_args[MAX_THREADS] = {0};
//     int chunk_size = batch_size / MAX_THREADS;
//     for(int e = 0; e < epochs; e++){
//         for (int i = 0; i < MAX_THREADS; i++) {
//             thread_args[i].input_layer = input_layer;
//             thread_args[i].output_layer = output_layer;
//             thread_args[i].samples = samples;
//             thread_args[i].labels = labels;
//             thread_args[i].l = nlayers-1;
//             thread_args[i].nlayers = nlayers;
//             thread_args[i].batch_size = chunk_size;
//             thread_args[i].batch_start = i * chunk_size;

//             int ret = pthread_create(&threads[i], NULL, (void*)neural_network_back_propagation_batch_pthreads, (void*)&thread_args[i]);

//             if (ret) {
//                 printf("Error:unable to create thread, %d\n", ret);
//                 exit(-1);
//             }
//         }

//         for(int i = 0; i < MAX_THREADS; i++) {
//             pthread_join(threads[i], NULL);
//         }
//     }
// }

 
void neural_network_train_batch(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int batch_size, int set_size, int epochs, float** samples, float** labels, uint cost_function, float learning_rate){
    printf("MINI-BATCH started:\n");
    for (int e = 0; e < epochs; e++){
        printf("epoch %d", e);
        float correct = 0;
        int inc = batch_size;
        for(int i = 0; i < set_size; i+=inc){
            correct += neural_network_back_propagation_batch(output_layer, input_layer, nlayers-1, nlayers, batch_size, samples + i, labels + i, cost_function, learning_rate);
            if(i+batch_size > set_size) inc = set_size - i;
        }
        correct /= set_size;
        printf(" completed\n");
        printf("Training accuracy: %f\n", correct);
        printf("Output Layer Activations:\n");
        print_arr(output_layer->a, output_layer->na);
        two_array_shuffle(samples, labels, set_size);
    }
}

float neural_network_predict(struct neuron_layer* input_layer, float* sample, float* label){
    struct neuron_layer* output_layer = neural_network_forward_pass_inputs(input_layer, sample);
    int actual = max_array_index(label, 10);
    int pred = max_array_index(output_layer->a, 10);
    return (float)(actual == pred);
}


void neural_network_train_stochastic(struct neuron_layer* input_layer, struct neuron_layer* output_layer, int nlayers, int set_size, int epochs, float** samples, float** labels, uint cost_function, float learning_rate){
    printf("SGD started:\n");
    for (int e = 0; e < epochs; e++){
        float correct = 0;
        printf("epoch %d", e);
        for (int i = 0; i < set_size; i++){
            // neural_network_forward_pass_inputs(input_layer, samples[i]);
            correct += neural_network_predict(input_layer, samples[i], labels[i]);
            neural_network_back_propagation_stochastic(output_layer, labels[i], nlayers-1, nlayers, cost_function, learning_rate);
        }
        correct /= set_size;
        printf(" completed\n");
        printf("Training accuracy: %f\n", correct);
        printf("Output Layer Activations:\n");
        print_arr(output_layer->a, output_layer->na);
        two_array_shuffle(samples, labels, set_size);
    }

}



float neural_network_test(struct neuron_layer* input_layer, struct neuron_layer* output_layer, float** samples, float** labels, int test_size){
   int correct = 0;
   for(int i = 0; i < test_size; i++){
        neural_network_forward_pass_inputs(input_layer, samples[i]);
        int digit = max_array_index(labels[i], 10);
        int pred = max_array_index(output_layer->a, 10);
        if(digit == pred) correct++;
   }
   return ((float)correct / (float)test_size);
}


char* neural_network_info_string(char* str, int* dims, int n){
	char* start = str;
    char* buf = str;
	buf += sprintf(buf, "MLP ");
	for(int i = 0; i < n-1; i++){
		buf += sprintf(buf, "%dx", dims[i]);
	}
	buf += sprintf(buf, "%d\n", dims[n-1]);
	buf += sprintf(buf, "Activations: %d ReLUs + Softmax\n", n-2);
    buf += sprintf(buf, "Loss: Cross Entropy\n");
    return start;
}