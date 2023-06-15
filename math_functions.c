#pragma once
#include "math_functions.h"


float rand_gauss(float mean, float sd) { // Box-Muller transform
    float u1, u2; // u1,u2 ~ U(0,1), [0,1]
    float x, y; // x,y ~ N(mean,sd)
    float r, theta;

    u1 = (float)rand() / (float)RAND_MAX;
    u2 = (float)rand() / (float)RAND_MAX;
    r = sqrt(-2.0 * log(u1)); 
    theta = 2.0 * M_PI * u2;

    x = r*cos(theta);
    y = r*sin(theta);

    x = sd*x + mean;
    y = sd*y + mean;
    //printf("Rand Gauss: u1:%f u2:%f r:%f theta:%f x:%f y:%f\n\n", u1, u2, r, theta, x, y);
    return ((float)rand() / (float)RAND_MAX > 0.5) ? x : y;
}

float* rand_gauss_weights(int nweights, float mean, float sd){
    float* rw = (float*)malloc(nweights*sizeof(float));
    for(int i = 0; i < nweights; i++){
        rw[i] = rand_gauss(mean, sd);
    }
    return rw;
}

float least_squares(float a, float y){
    return (a-y)*(a-y);
}
float least_squares_derivative(float a, float y){
    return 2*(a-y);
}

float cross_entropy_softmax_derivative(float a, float y){
    return (y-a);
}

float sum_func(float(*f)(float, float), float* a, float* y, int n){
    float sum = 0;
    for(int i = 0; i<n; i++){
        sum+=f(a[i],y[i]);
    }
    return sum;    
}

float* element_wise_func2(float(*f)(float, float), float* a, float* y, float* res, int n){
    for(int i = 0; i<n; i++){
        // printf("passed");
        res[i] = f(a[i],y[i]);
        // printf("%f\n", res[i]);
    }
    return res;
}

float* element_wise_func(float(*f)(float), float* a, float* res, int n){
    for(int i = 0; i<n; i++){
        res[i] = f(a[i]);
    }
    return res;
}


float* element_wise_mut_func2(float(*f)(float, float), float* dest, float* src, int n){
    for(int i = 0; i<n; i++){
        dest[i] = f(dest[i],src[i]);
    }
    return dest;
}

float* element_wise_mut_func(float(*f)(float), float* dest, int n){
    for(int i = 0; i<n; i++){
        dest[i] = f(dest[i]);
    }
    return dest;
}
float* element_wise_mut_val_div(float val, float* dest, int n){
    for(int i = 0; i<n; i++){
        dest[i] /= val;
    }
    return dest;
}
float* element_wise_mut_val_sub(float val, float* dest, int n){
    for(int i = 0; i<n; i++){
        dest[i] -= val;
    }
    return dest;
}

float product(float x, float y){
    return x*y;
}
float quotient(float x, float y){
    return x/y;
}

// float* hadamard_product(float* x, float* y, float* z, int n){
//     return element_wise_func(product, x, y, z, n);
// }

float* matrix_multiply(float* x, int xrows, int xcols, float* y, int yrows, int ycols, float* dest){
    // ith row jth column
    // MxN . NxP
    // Dij = Σ_k XikYkj;
    int kmax = (xcols == yrows) ? xcols : 0;
    for(int i = 0; i < xrows; i++){
        for(int j = 0; j < ycols; j++){
            float sum = 0;
            for(int k = 0; k < kmax; k++){
                sum += x[i*xcols + k]*y[k*ycols + j];
            }
            dest[i*ycols + j] = sum;
        }
    }
    return dest;
}

float* matrix_multiply_transpose(float* x, int xrows, int xcols, float* y, int yrows, int ycols, float* dest){
    // First matrix is multiplied as if it was transposed
    // ith row jth column
    // ROWS X COLS
    // X: MxN . Y: QxP, D = MXP, X_ij = X[i*xcols + j], Y_ij = X[i*ycols + j], D_ij = D[i*ycols + j]
    // X_T NXM Y: QXP, D = NXP ... M=Q, X^T_ij = X[j*xrows + i], Y_ij = [i*ycols + j], D_ij = D[i*ycols + j]
    // D = X^T Y where D_ij = Σ_k X_ki Y_kj = Σ_k X^T_ik Y_kj

    int kmax = (xrows == yrows) ? xrows : 0;
    for(int i = 0; i < xcols; i++){
        for(int j = 0; j < ycols; j++){
            float sum = 0;
            for(int k = 0; k < kmax; k++){
                sum += x[k*xrows + i]*y[k*ycols + j];
            }
            dest[i*ycols + j] = sum;
        }
    }
    return dest;
}

float* hadamard_product(float* x, float* y, int n){ // mutates //dest*=source
    return element_wise_mut_func2(product, x, y, n);
}

float* cost_activation_gradient_mse(float* a, float* y, float* dest, int n){
    return element_wise_func2(least_squares_derivative, a, y, dest, n);
}
float* cost_activation_gradient_cel(float* a, float* y, float* dest, int n){
    return element_wise_func2(cross_entropy_softmax_derivative, a, y, dest, n);
}

// cost activation gradient batch

float relu(float x){
    return (x>0) ? x : 0;
}

float unity(float x){
    return 1;
}

float step(float x){
    return (x>0) ? 1 : 0;
}

float sigmoid(float x){
    return 1/(1+expf(-x));
}
float sigmoid_derivative(float x){
    return sigmoid(x)*(1-sigmoid(x));
}

float dot(float* x, float* y, int len){
    float sum = 0;
    for(int i = 0; i<len; i++) sum += x[i]*y[i];
    return sum;
}

float vec_mean(float* vec, int size){
    float sum = 0;
    for (int i = 0; i < size; i++){
        sum+=vec[i];
    }
    return sum/(float)size;
}

int max(int a, int b) {
    return a > b ? a : b;
}


float log_sum_exp(float a, float b) {
    if (a == b) {
        return a + log(2.0f);
    } else {
        float max_val = max(a, b);
        return max_val + log(expf(a - max_val) + expf(b - max_val));
    }
}

float safe_exp(float z) {
    float max_val = FLT_MAX;
    return (z > max_val) ? max_val : expf(z);
}

float cross_entropy_loss(float* p, float* y){
//    Eigen::VectorXf lnp = p.array().log();
//    float loss = (y.array() * lnp.array()).sum();
    float loss = 0;
    return loss;
}

float stddev_vec(float* vec){
//    float std_dev = sqrt((vec.array() - vec.mean()).square().sum()/(vec.size()-1));
    float std_dev = 0;
    return std_dev;
}


// float* softmax(flo)
float sumf(float* x, int n){
    float sum = 0;
    for(int i = 0; i<n; i++){
        sum+=x[i];
    }
    return sum;    
}

float max_coeff_arr(float* x, int n){
    float max = x[0];
    for(int i = 1; i < n; i++){
        if(x[i]>max) max = x[i];
    }
    return max;
}

float* softmax(float* x, int n){
    // element_wise_mut_func(exp, x, n);
    float expsum = sumf(x, n);
    return element_wise_mut_val_div(expsum, x, n);
}
