#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/s8.h"
#include "weights/b8.h"

//hls-fpga-machine-learning insert layer-config
// Hidden_layer_2
struct config2 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 640;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef hidden_layer_2_bias_t bias_t;
    typedef hidden_layer_2_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// Hidden_layer_2_relu
struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// Hidden_layer_3
struct config4 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 50;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef hidden_layer_3_bias_t bias_t;
    typedef hidden_layer_3_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// Hidden_layer_3_relu
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// Output_layer
struct config6 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 10;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef output_layer_bias_t bias_t;
    typedef output_layer_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// Output_layer_softmax
struct softmax_config7 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::latency;
    typedef ap_fixed<18,8> exp_table_t;
    typedef ap_fixed<18,8> inv_table_t;
};

// batch_normalization_1
struct config8 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_1_bias_t bias_t;
    typedef batch_normalization_1_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};


#endif
