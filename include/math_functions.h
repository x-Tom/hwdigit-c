#include <math.h>
#include <stdlib.h>
#include <float.h>

typedef unsigned uint;
typedef float*(*cost_act_grad)(float*,float*,float*,int);


float rand_gauss(float mean, float sd);
float* rand_gauss_weights(int nweights, float mean, float sd);
float least_squares(float a, float y);
float least_squares_derivative(float a, float y);
float sum_func(float(*f)(float, float), float* a, float* y, int n);
float* element_wise_func2(float(*f)(float, float), float* a, float* y, float* res, int n);
float* element_wise_func(float(*f)(float), float* a, float* res, int n);
float* element_wise_mut_func2(float(*f)(float, float), float* dest, float* src, int n);
float* element_wise_mut_func(float(*f)(float), float* dest, int n);
float* element_wise_mut_val_div(float val, float* dest, int n);
float* element_wise_mut_val_sub(float val, float* dest, int n);
float product(float x, float y);
float* matrix_multiply(float* x, int xrows, int xcols, float* y, int yrows, int ycols, float* dest);
float* matrix_multiply_transpose(float* x, int xrows, int xcols, float* y, int yrows, int ycols, float* dest);
float* hadamard_product(float* x, float* y, int n); // mutates //dest*=source
float* cost_activation_gradient_mse(float* a, float* y, float* dest, int n);
float relu(float x);
float step(float x);
float unity(float x);
float sigmoid(float x);
float sigmoid_derivative(float x);
float dot(float* x, float* y, int len);
float vec_mean(float* vec, int size);
int max(int a, int b);
float log_sum_exp(float a, float b);
float safe_exp(float z);
float cross_entropy_loss(float* p, float* y);
float stddev_vec(float* vec);
float sumf(float* x, int n);
float max_coeff_arr(float* x, int n);
float* softmax(float* x, int n);
