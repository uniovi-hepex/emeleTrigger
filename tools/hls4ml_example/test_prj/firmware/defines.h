#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 64
#define N_LAYER_2 10
#define N_LAYER_4 5
#define N_LAYER_6 2

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_int<2> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> hidden_layer_2_weight_t;
typedef ap_fixed<16,6> hidden_layer_2_bias_t;
typedef ap_fixed<16,6> hidden_layer_2_relu_default_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> hidden_layer_3_weight_t;
typedef ap_fixed<16,6> hidden_layer_3_bias_t;
typedef ap_fixed<16,6> hidden_layer_3_relu_default_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> output_layer_weight_t;
typedef ap_fixed<16,6> output_layer_bias_t;
typedef ap_fixed<16,6> output_layer_softmax_default_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<16,6> batch_normalization_1_scale_t;
typedef ap_fixed<16,6> batch_normalization_1_bias_t;

#endif
