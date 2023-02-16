/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#define USE_INVERTED_R true

#include "dataset.cuh"
#include "mic.cuh"
#include "nep3.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"

#include <stdio.h>

static __global__ void gpu_find_neighbor_list(
  const int N,
  const int* Na,
  const int* Na_sum,
  const float rc2_radial,
  const float rc2_angular,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* x12_angular,
  float* y12_angular,
  float* z12_angular)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < rc2_radial) {
              NL_radial[count_radial * N + n1] = n2;
              x12_radial[count_radial * N + n1] = x12;
              y12_radial[count_radial * N + n1] = y12;
              z12_radial[count_radial * N + n1] = z12;
              count_radial++;
            }
            if (distance_square < rc2_angular) {
              NL_angular[count_angular * N + n1] = n2;
              x12_angular[count_angular * N + n1] = x12;
              y12_angular[count_angular * N + n1] = y12;
              z12_angular[count_angular * N + n1] = z12;
              count_angular++;
            }
          }
        }
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
  }
}

static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::DNN dnnmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_NUM_N] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      // CHANGE: Using 1/r instead of r
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      if (USE_INVERTED_R) {
        d12 = 1 / d12;
      }
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, USE_INVERTED_R);
      int t2 = g_type[n2];
      float fn12[MAX_NUM_N];
      if (paramb.version == 2) {
        find_fn(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float c = (paramb.num_types == 1)
                      ? 1.0f
                      : dnnmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          q[n] += fn12[n] * c;
        }
      } else {
        find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gn12 += fn12[k] * dnnmb.c[c_index];
          }
          q[n] += gn12;
        }
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = q[n];
    }
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::DNN dnnmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = n1 + N * i1;
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        // INVERTED_R: ignored for angular ones
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
        int t2 = g_type[n2];
        if (paramb.version == 2) {
          float fn;
          find_fn(n, paramb.rcinv_angular, d12, fc12, fn);
          fn *=
            (paramb.num_types == 1)
              ? 1.0f
              : dnnmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          accumulate_s(d12, x12, y12, z12, fn, s);
        } else {
          float fn12[MAX_NUM_N];
          find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * dnnmb.c[c_index];
          }
          accumulate_s(d12, x12, y12, z12, gn12, s);
        }
      }
      if (paramb.num_L == paramb.L_max) {
        find_q(paramb.n_max_angular + 1, n, s, q);
      } else if (paramb.num_L == paramb.L_max + 1) {
        find_q_with_4body(paramb.n_max_angular + 1, n, s, q);
      } else {
        find_q_with_5body(paramb.n_max_angular + 1, n, s, q);
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.num_L; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

NEP3::NEP3(
  char* input_dir,
  Parameters& para,
  int N,
  int N_times_max_NN_radial,
  int N_times_max_NN_angular,
  int version)
{
  paramb.version = version;
  paramb.rc_radial = para.rc_radial;
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rc_angular = para.rc_angular;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  // dnnmb.dim = para.dim;
  dnnmb.n_layers = para.num_layers + 2;

  // We set topology as a const variable, so we initialize it
  // separately
  int* topology = new int[para.num_layers + 2];
  topology[0] = para.dim;
  for (int i = 0; i < para.num_layers; i++) {
    topology[i + 1] = para.hidden_sizes[i];
  }
  // Output size
  topology[para.num_layers + 1] = 1;
  dnnmb.topology = topology;

  int num_weights = 0;
  int num_biases = 0;
  for (int i = 0; i < para.num_layers + 1; i++) {
    num_weights += dnnmb.topology[i] * dnnmb.topology[i + 1];
    num_biases += dnnmb.topology[i + 1];
  }
  // // Weights that correspond to each output layer
  // dnnmb.weights = new float[num_weights];
  // // Biases that correspond to each output layer
  // dnnmb.biases = new float[num_biases];
  dnnmb.num_weights = num_weights;
  dnnmb.num_biases = num_biases;
  // dnnmb.hidden_sizes = para.hidden_sizes;
  paramb.num_types = para.num_types;
  dnnmb.num_para = para.number_of_variables;
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.num_L = paramb.L_max;
  
  if (version == 3) {
    if (para.L_max_4body == 2) {
      paramb.num_L += 1;
    }
    if (para.L_max_5body == 1) {
      paramb.num_L += 1;
    }
  }
  paramb.dim_angular = (para.n_max_angular + 1) * paramb.num_L;

  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

  zbl.enabled = para.enable_zbl;
  zbl.rc_inner = para.zbl_rc_inner;
  zbl.rc_outer = para.zbl_rc_outer;
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    zbl.atomic_numbers[n] = para.atomic_numbers[n];
  }

  // Added by Michael Fatemi, 2022 November 12
  coulomb.enabled = para.enable_coulomb;
  coulomb.alpha = para.coulomb_alpha;
  coulomb.epsilon = para.coulomb_epsilon;
  for (int n = 0; n < para.coulomb_charges.size(); ++n) {
    coulomb.charges[n] = para.coulomb_charges[n];
  }

  nep_data.NN_radial.resize(N);
  nep_data.NN_angular.resize(N);
  nep_data.NL_radial.resize(N_times_max_NN_radial);
  nep_data.NL_angular.resize(N_times_max_NN_angular);
  nep_data.x12_radial.resize(N_times_max_NN_radial);
  nep_data.y12_radial.resize(N_times_max_NN_radial);
  nep_data.z12_radial.resize(N_times_max_NN_radial);
  nep_data.x12_angular.resize(N_times_max_NN_angular);
  nep_data.y12_angular.resize(N_times_max_NN_angular);
  nep_data.z12_angular.resize(N_times_max_NN_angular);
  nep_data.descriptors.resize(N * dnnmb.topology[0]);
  nep_data.Fp.resize(N * dnnmb.topology[0]);
  nep_data.sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
  nep_data.parameters.resize(dnnmb.num_para);
}

void NEP3::update_potential(const float* parameters, DNN& dnn)
{
  dnn.weights = parameters;
  dnn.biases = parameters + dnn.num_weights;
  dnn.c = parameters + dnn.num_weights + dnn.num_biases;
}

static void __global__ find_max_min(const int N, const float* g_q, float* g_q_scaler)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  s_max[tid] = -1000000.0f; // a small number
  s_min[tid] = +1000000.0f; // a large number
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      float q = g_q[m];
      if (q > s_max[tid]) {
        s_max[tid] = q;
      }
      if (q < s_min[tid]) {
        s_min[tid] = q;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) {
        s_max[tid] = s_max[tid + offset];
      }
      if (s_min[tid] > s_min[tid + offset]) {
        s_min[tid] = s_min[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_q_scaler[bid] = min(g_q_scaler[bid], 1.0f / (s_max[0] - s_min[0]));
  }
}

static __device__ void apply_affine(
  const int input_dim,
  const int output_dim,
  float* input,
  const float* weight,
  const float* bias,
  // used to return results
  float* output,
  bool apply_activation
) {
  for (int output_i = 0; output_i < output_dim; output_i++) {
    float weighted_input = bias[output_i];
    for (int input_i = 0; input_i < input_dim; input_i++) {
      weighted_input += weight[output_i * input_dim + input_i] * input[input_i];
    }
    if (apply_activation) {
      output[output_i] = tanh(weighted_input);
    } else {
      output[output_i] = weighted_input;
    }
  }
}

static __device__ void affine_backward(
  const int input_dim,
  const int output_dim,
  bool apply_activation,
  const float* input,
  const float* weight,
  const float* bias,
  const float* output,
  // set by backprop
  float* input_grad,
  float* weight_grad,
  float* bias_grad,
  const float* output_grad // ,
  // const float* expected_energy
) {
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;

  // if (n1 == 0) {
  //   printf("output_grad[0] = %f\n", output_grad[0]);
  // }

  /*
  output[o] = [input[i] * weight[o][i] across all i] + bias[o]
  input_grad[i] = weight[o][i] * output_grad[o] across all outputs
  weight_grad[o][i] = input[i] * output_grad[o]
  bias_grad[o] = output_grad[o]
  */
  // atomicAdd ensures that we can do this across examples
  // without setting bad values (such as reading A in 1,
  // reading A in 2, writing A + X with 1, writing A + Y with 2,
  // resulting in A + Y instead of A + X + Y)
  for (int i = 0; i < input_dim; i++) {
    for (int o = 0; o < output_dim; o++) {
      float grad_change = weight[o * input_dim + i] * output_grad[o] *
        (apply_activation ? (1 - output[o] * output[o]) : 1);

      // atomicAdd(&input_grad[i], grad_change);
      
      input_grad[i] += grad_change;
    }
  }
  if (weight_grad != nullptr) {
    for (int i = 0; i < input_dim; i++) {
      for (int o = 0; o < output_dim; o++) {
        float grad_change = input[i] * output_grad[o] *
          (apply_activation ? (1 - output[o] * output[o]) : 1);

        atomicAdd(&weight_grad[o * input_dim + i], grad_change);

        // weight_grad[o * input_dim + i] += grad_change;
      }
    }
  }
  if (bias_grad != nullptr) {
    for (int o = 0; o < output_dim; o++) {
      float grad_change = output_grad[o] * 
        (apply_activation ? (1 - output[o] * output[o]) : 1);

      atomicAdd(&bias_grad[o], grad_change);

      // if ((grad_change < 0) && !apply_activation) {
      //   printf("grad_change < 0; output[%d] = %f; output_grad[%d] = %f\n", o, output[o], o, output_grad[o]);
      // }

      // if (o == 0 && output_dim == 4 && output_grad[0] < 0 && output[0] != 31) {
      //   printf("bias_grad[0] += %f; output_grad[0] = %f; output[0] = %f; expected_energy = %f\n", grad_change, output_grad[0], output[0], expected_energy[0]);
      // }

      // bias_grad[o] += grad_change;
    }
  }
}

static __device__ float abs_(float x) {
  return x < 0 ? -x : x;
}

static __device__ void huber_grad(float& x) {
  if (x > 1) {
    x = 1;
  } else if (x < -1) {
    x = -1;
  }
}

/*
Deep neural network function, with support for backpropagation. The "force" outputs are
actually proxies that are used relative to the descriptors to calculate the actual force.
The code to recover the force is written by the creators of the library and I am not familiar
with how it works.

To calculate a gradient, pass a value to `energy_ref_dev`. If you do this, the method will
assume that you have provided an appropriately-sized `gradient` 2D pointer. This method only
cares about loss with respect to force and energy; regularization techniques like L1 and L2
regularization can be added manually outside of this method.

This function is called in apply_dnn.

apply_dnn_layers(dnnmb.n_layers, dev_topology, dnnmb.weights, dnnmb.biases, q, output, energy_ref_dev, gradient);
*/
static __device__ void apply_dnn_layers(
  const int N,
  const int n_layers,
  const int* topology,
  const float* weights,
  const float* biases,
  float* q,
  float* output,
  // Only contains a single value
  // const float* energy_ref,
  float* input_grad
) {
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  
  float activations[5][30];
  for (int i = 0; i < topology[0]; i++) {
    activations[0][i] = q[i];
  }
  int cumulative_weights = 0;
  int cumulative_biases = 0;
  for (int layer_i = 0; layer_i < n_layers - 1; layer_i++) {
    int input_size = topology[layer_i];
    int output_size = topology[layer_i + 1];
    apply_affine(
      input_size,
      output_size,
      activations[layer_i],
      weights + cumulative_weights,
      biases + cumulative_biases,
      activations[layer_i + 1],
      layer_i + 1 < n_layers - 1 // don't apply activation on last layer
    );
    cumulative_weights += input_size * output_size;
    cumulative_biases += output_size;
  }

  __syncthreads();

  for (int i = 0; i < topology[n_layers - 1]; i++) {
    output[i] = activations[n_layers - 1][i];
  }

  // if (energy_ref == nullptr) {
  //   return;
  // }

  float activation_grad[5][30];

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 30; j++) {
      activation_grad[i][j] = 0;
    }
  }

  // // Backpropagation
  // // Gradient of loss w.r.t. each output value goes here
  // float expected_energy = energy_ref[0];
  // // Use the real values for this later on
  // float expected_fx = 0;
  // float expected_fy = 0;
  // float expected_fz = 0;
  // // float expected_fx = expected_output[1];
  // // float expected_fy = expected_output[2];
  // // float expected_fz = expected_output[3];

  // Loss is 0.5 x sqrt(MSE) = 0.5 x (y - yhat)^2 = y - yhat.
  int weight_grad_ptr = cumulative_weights;
  int bias_grad_ptr = cumulative_biases;

  // float NF = (float) N;

  // MSE
  // activation_grad[n_layers - 1][0] = -(expected_energy - output[0]); // / NF;
  // activation_grad[n_layers - 1][1] = -(expected_fx - output[1]); // / NF;
  // activation_grad[n_layers - 1][2] = -(expected_fy - output[2]); // / NF;
  // activation_grad[n_layers - 1][3] = -(expected_fz - output[3]); // / NF;

  // // MAE
  // float s1 = (output[0] > expected_energy) ? 1 : -1;
  // float s2 = (output[1] > expected_fx) ? 1 : -1;
  // float s3 = (output[2] > expected_fy) ? 1 : -1;
  // float s4 = (output[3] > expected_fz) ? 1 : -1;

  // activation_grad[n_layers - 1][0] = s1 / NF;
  // activation_grad[n_layers - 1][1] = s2 / NF;
  // activation_grad[n_layers - 1][2] = s3 / NF;
  // activation_grad[n_layers - 1][3] = s4 / NF;

  // Calculate the gradient with respect to energy
  activation_grad[n_layers - 1][0] = 1;

  // Huber loss (turns into MAE after a cutoff point)
  // for (int i = 0; i < 4; i++) {
  //   huber_grad(activation_grad[n_layers - 1][i]);
  //   // if (n1 == 0) {
  //   //   printf("activation_grad[%d][%d] = %f\n", n_layers - 1, i, activation_grad[n_layers - 1][i]);
  //   // }
  // }

  // if (n1 < 10) {
  //   printf("pred=%f true=%f grad=%f\n", output[0], expected_energy, activation_grad[n_layers - 1][0]);
  // }

  // For every pair of layers, where layer_i is the earlier layer
  // and layer_i + 1 is the next layer:
  for (int layer_i = n_layers - 2; layer_i >= 0; layer_i--) {
    int input_dim = topology[layer_i];
    int output_dim = topology[layer_i + 1];
    weight_grad_ptr -= input_dim * output_dim;
    bias_grad_ptr -= output_dim;

    // if (n1 == 0) {
    //   printf("<< Layer: %d >>\n", layer_i);
    // }

    // if (n1 == 0) {
    //   printf("activation_grad[layer_i + 1][0] = %f\n", activation_grad[layer_i + 1][0]);
    // }

    affine_backward(
      input_dim,
      output_dim,
      layer_i + 1 < n_layers - 1, // don't apply activation on last layer
      activations[layer_i],
      weights + weight_grad_ptr,
      biases + bias_grad_ptr,
      activations[layer_i + 1],
      activation_grad[layer_i],
      nullptr,
      nullptr,
      // weight_grad + weight_grad_ptr,
      // bias_grad + bias_grad_ptr,
      activation_grad[layer_i + 1] // ,
      // energy_ref
    );

    // if (n1 == 0) {
    //   printf("activation_grad[layer_i + 1][0] = %f\n", activation_grad[layer_i + 1][0]);
    // }

    __syncthreads();

    // if (n1 == 0) {
    //   if (layer_i + 1 == n_layers - 1) {
    //     printf("final activation: %f; final activation grad: %f; final activation bias: %f; final activation bias grad: %f\n", activations[n_layers - 1][0], activation_grad[n_layers - 1][0], biases[bias_grad_ptr + 0], bias_grad[bias_grad_ptr + 0]);
    //   }
    //   // printf("")
    //   // printf("<weight_grad[%d][0] weight[%d][0]>: <%.2f %.2f>\n", layer_i, layer_i, (weight_grad + weight_grad_ptr)[0], (weights + weight_grad_ptr)[0]);
    //   // printf("<activation_grad[%d + 1][0] activation[%d + 1][0]>: <%f %f>\n", layer_i, layer_i, (activation_grad[layer_i + 1])[0], (activations[layer_i + 1])[0]);
    //   // printf("<bias_grad[%d][0] bias[%d][0]>: <%f %f>\n", layer_i, layer_i, (bias_grad + bias_grad_ptr)[0], (biases + bias_grad_ptr)[0]);
    // }

    // if (n1 == 0) {
    //   printf("activation_grad[layer_i=%d][0] = %f\n", layer_i, activation_grad[layer_i][0]);
    // }
  }

  for (int i = 0; i < topology[0]; i++) {
    input_grad[i] = activation_grad[0][i];
  }
  // if (n1 == 0) { printf("\n"); }
}

static __global__ void apply_dnn(
  const int N,
  const NEP3::ParaMB paramb,
  const NEP3::DNN dnnmb,
  const int* dev_topology,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp // ,
  // const float* energy_ref,
  // float* dev_weight_grad,
  // float* dev_bias_grad
) {
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    // if (n1 == 0) {
    //   printf("N: %d\n", N);
    // }
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < dev_topology[0]; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    float Fp[MAX_DIM] = {0.0f};
    float output[1];
    apply_dnn_layers(
      N,
      dnnmb.n_layers,
      dev_topology,
      dnnmb.weights,
      dnnmb.biases,
      q,
      output,
      // energy_ref ? (&energy_ref[n1]) : nullptr,
      Fp
    );
    g_pe[n1] = output[0];
    // if (n1 == 0) {
    //   printf("output energy: %f\n", output[0]);
    // }
    for (int d = 0; d < dev_topology[0]; ++d) {
      // TODO: replace with real force
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void apply_ann(
  const int N,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0, annmb.b0, annmb.w1, annmb.b1, q, F, Fp);
    g_pe[n1] = F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void zero_force(const int N, float* g_fx, float* g_fy, float* g_fz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
  }
}

// Written by Michael Fatemi, 2022 November 15
/*
Returns the magnitude of the force vector attributable to the Coulomb force.

From equation (18) of https://aip.scitation.org/doi/10.1063/1.2206581
"Is Ewald Summation Still Necessary?" by Fennell and Gezelter, 2006

force = q_i * q_j *
  (erfc(ar) / r^2 + 2a/(pi^1/2) * exp(-(a^2 * r^2)) / r
  - erfc(arc) / rc^2 + 2a/(pi^1/2) * exp(-(a^2 * rc^2)) / rc)

erfc(ar) / r^2 + 2a/(pi^1/2) * exp(-(a^2 * r^2)) / r

[Deprecated:]
If r=2ang, and eps=38, then in eV the potential energy should be 13.6/(38*2/0.52918)=0.1 eV
1/r --> 13.6/(epsilon * r/0.52918)
1/r^2 --> 13.6/(epsilon * r/0.52918 * r)

[Revised:]
If you work in SI units 1/4pi epsilon_0 = 9 10^9
Electron charge q=1.6 10^-19 (to convert joule to eV, one factor of q drops out)
 
Potential = 1/(4 pi epsilon_0 epsilon) *q^2/r = 9 10^9/38 *1.6 10^-19 / 2 10^-10 =9*1.6/(2*38)=  0.19 eV

1/r --> 9 * 1.6 / (r * epsilon)

*/
static __device__ float _coulomb_force_part(float r, float alpha, float epsilon) {
  // return (13.6 / (epsilon * r / 0.52918 * r)) * (erfc(alpha * r) + 2 * alpha / sqrt(PI) * r * exp(-(alpha * alpha * r * r)));
  return (9 * 1.6 / (epsilon * r * r)) * (erfc(alpha * r) + 2 * alpha / sqrt(PI) * r * exp(-(alpha * alpha * r * r)));
}

// Added by Michael Fatemi, 2022 November 12
/*
Accumulates Coulomb force. Normalizes such that the potential at r_cutoff is 0.

Parameters:
 - t{1, 2} (int): Type (atomic number) of first and second atom.
 - r12 (float*): Vector pointing from first atom to second atom.
 - coulomb (NEP3::Coulomb): Used here for the value of epsilon, alpha (the damping factor),
   and the charges of each atom type.
 - rc_radial (float): The radial cutoff (used to normalize the potential).
 - f12 (float*): Vector containing force of first atom on second atom.
*/
static __device__ void add_coulomb_force(
  float q1,
  float q2,
  float* r12,
  NEP3::Coulomb coulomb,
  float rc_radial,
  float* f12)
{
  float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

  /*
  const float sqrt_pi = sqrt(PI);
  float alpha = coulomb.alpha;
  float mag_r = (erfc(alpha * d12) / (d12 * d12) + 2 * alpha / sqrt_pi * exp(-(alpha * alpha * d12 * d12)) / d12);
  float mag_rc = (erfc(alpha * rc_radial) / (rc_radial * rc_radial) + 2 * alpha / sqrt_pi * exp(-(alpha * alpha * rc_radial * rc_radial)) / rc_radial);
  float coulomb_constant = 1 / (4 * PI * coulomb.epsilon);
  float mag = q1 * q2 * (mag_r - mag_rc) * coulomb_constant;
  */
  float mag = q1 * q2 * (_coulomb_force_part(d12, coulomb.alpha, coulomb.epsilon) - _coulomb_force_part(rc_radial, coulomb.alpha, coulomb.epsilon));

  // Add force in direction of r21 (reverse, so positive magnitude = repulsion)
  f12[0] += -mag * r12[0] / d12;
  f12[1] += -mag * r12[1] / d12;
  f12[2] += -mag * r12[2] / d12;
}

// Added by Michael Fatemi, 2022 November 15
/*
From equation (16) of https://aip.scitation.org/doi/10.1063/1.2206581
"Is Ewald Summation Still Necessary?" by Fennell and Gezelter, 2006

energy = q_i * q_j * (erfc(ar) / r - erfc(arc) / rc)

[Deprecated:]
If r=2ang, and eps=38, then in eV the potential energy should be 13.6/(38*2/0.52918)=0.1 eV
1/r --> 13.6/(epsilon * r/0.52918)
1/r^2 --> 13.6/(epsilon * r/0.52918 * r)

[Revised:]
If you work in SI units 1/4pi epsilon_0 = 9 10^9
Electron charge q=1.6 10^-19 (to convert joule to eV, one factor of q drops out)
 
Potential = 1/(4 pi epsilon_0 epsilon) *q^2/r = 9 10^9/38 *1.6 10^-19 / 2 10^-10 =9*1.6/(2*38)=  0.19 eV

1/r --> 9 * 1.6 / (r * epsilon)
*/
static __device__ float _coulomb_potential_part(float r, float alpha, float epsilon) {
  return 9 * 1.6 / (r * epsilon) * erfc(r * alpha);
}

// Added by Michael Fatemi, 2022 November 15
/*
Accumulates Coulomb potential. Normalizes such that the potential at r_cutoff is 0.

Parameters:
 - t{1, 2} (int): Type (atomic number) of first and second atom.
 - r12 (float*): Vector pointing from first atom to second atom.
 - coulomb (NEP3::Coulomb): Used here for the value of epsilon, alpha (the damping factor),
   and the charges of each atom type.
 - rc_radial (float): The radial cutoff (used to normalize the potential).
 - f12 (float&): Reference to variable containing potential energy of first
*/
static __device__ void add_coulomb_potential(
  int t1,
  int t2,
  float* r12,
  NEP3::Coulomb coulomb,
  float rc_radial,
  float& energy)
{
  float q1 = coulomb.charges[t1];
  float q2 = coulomb.charges[t2];

  float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

  energy += q1 * q2 * (_coulomb_potential_part(d12, coulomb.alpha, coulomb.epsilon) - _coulomb_potential_part(rc_radial, coulomb.alpha, coulomb.epsilon));
}

// static __global__ void calculate_charges(
//   const int N,
//   const int* g_NN,
//   const int* g_NL,
//   const int* __restrict__ g_type,
//   const float* base_charges,
//   const int oxygen_type,
//   float* output_charges
// ) {
//   int n1 = threadIdx.x + blockIdx.x * blockDim.x;
//   if (n1 >= N) {
//     return;
//   }

//   if (n1 == 0) {
//     printf("Base charges: ");
//     for (int i = 0; i < 2; i++) {
//       printf("%f ", dnnmb.charges[i]);
//     }
//     printf("\n");
//   }

//   int t1 = g_type[n1];
//   if (t1 != oxygen_type) {
//     output_charges[n1] = dnnmb.charges[t1];
//     return;
//   }
//   int neighbor_number = g_NN[n1];
//   float total_charge = 0;
//   int count = 0;
//   for (int i = 0; i < neighbor_number; i++) {
//     int index = i * N + n1;
//     int n2 = g_NL[index];
//     int t2 = g_type[n2];
//     if (t2 != oxygen_type) {
//       total_charge += dnnmb.charges[t2];
//       count += 1;
//     }
//   }
//   if (count == 0) {
//     // Default charge of oxygen
//     output_charges[n1] = -2;
//   } else {
//     output_charges[n1] = -total_charge/count;
//   }
// }

// Accumulates force and energy interactions
static __global__ void accumulate_radial_interactions(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::DNN dnnmb,
  const NEP3::Coulomb coulomb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe //,
  // float* coulomb_forces,
  // Calculated charge of each atom
  // const float* charges
)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int t1 = g_type[n1];
    float total_coulomb[3] = {0.0f, 0.0f, 0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      if (paramb.version == 2) {
        if (USE_INVERTED_R) {
          find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, 1 / d12, fc12, fcp12, fn12, fnp12);
        } else {
          find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        }
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          // Change: Add chain rule for d/dr E(1/r) -> d/d(1/r) E(1/r) x d/dr 1/r
          // Multiply by -1/r^2
          float tmp12 = g_Fp[n1 + n * N] * fnp12[n];
          if (USE_INVERTED_R) {
            tmp12 *= -1/(d12 * d12);
          }
          tmp12 *= (paramb.num_types == 1)
                     ? 1.0f
                     : dnnmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * (r12[d] * d12inv);
          }
        }
      } else {
        // if (n1 == 0) {
        //   printf("1/d12: %f\n", d12inv);
        // }
        // Change: Use 1/r
        if (USE_INVERTED_R) {
          find_fn_and_fnp(paramb.basis_size_radial, paramb.rcinv_radial, 1 / d12, fc12, fcp12, fn12, fnp12);
        } else {
          find_fn_and_fnp(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        }
        // paramb.basis_size_radial, paramb.rcinv_radial, 1 / d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gnp12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gnp12 += fnp12[k] * dnnmb.c[c_index];
          }
          // Account for chain rule, as mentioned above          
          float tmp12 = g_Fp[n1 + n * N] * gnp12;
          if (USE_INVERTED_R) {
            tmp12 *= -1/(d12 * d12);
          }
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * (r12[d] * d12inv);
          }
        }
        // if (n1 == 0) {
        //   printf("fx fy fz: %.3f %.3f %.3f\n", f12[0], f12[1], f12[2]);
        // }
      }

      // Added by Michael Fatemi, 2022 November 12
      // Integrate Coulomb force calculation with radial force function
      // if (coulomb.enabled) {
      //   float fx = f12[0];
      //   float fy = f12[1];
      //   float fz = f12[2];
      //   int q1 = charges[n1];
      //   int q2 = charges[n2];
      //   add_coulomb_force(q1, q2, r12, coulomb, paramb.rc_radial, f12);
      //   total_coulomb[0] += f12[0] - fx;
      //   total_coulomb[1] += f12[1] - fy;
      //   total_coulomb[2] += f12[2] - fz;
      //   // n1 refers to the current structure
      //   add_coulomb_potential(q1, q2, r12, coulomb, paramb.rc_radial, g_pe[n1]);
      // }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];

    }

    // Logged by us
    // coulomb_forces[n1] = total_coulomb[0];
    // coulomb_forces[n1 + N] = total_coulomb[1];
    // coulomb_forces[n1 + N * 2] = total_coulomb[2];
    // coulomb_forces[n1] = sqrt((total_coulomb[0] * total_coulomb[0]) + (total_coulomb[1] * total_coulomb[1]) + (total_coulomb[2] * total_coulomb[2]));

    g_virial[n1] = s_virial_xx;
    g_virial[n1 + N] = s_virial_yy;
    g_virial[n1 + N * 2] = s_virial_zz;
    g_virial[n1 + N * 3] = s_virial_xy;
    g_virial[n1 + N * 4] = s_virial_yz;
    g_virial[n1 + N * 5] = s_virial_zx;
  }
}

static __global__ void find_force_angular(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::DNN dnnmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {

    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      float f12[3] = {0.0f};
      // Don't use the inverted one here.
      // Angular descriptor calculations are kinda diff.
      float used_d12 = d12; // (USE_INVERTED_R) ? (1 / d12) : d12;

      if (paramb.version == 2) {
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          float fn;
          float fnp;
          find_fn_and_fnp(n, paramb.rcinv_angular, used_d12, fc12, fcp12, fn, fnp);
          const float c =
            (paramb.num_types == 1)
              ? 1.0f
              : dnnmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          fn *= c;
          fnp *= c;
          accumulate_f12(n, paramb.n_max_angular + 1, used_d12, r12, fn, fnp, Fp, sum_fxyz, f12);
        }
      } else {
        float fn12[MAX_NUM_N];
        float fnp12[MAX_NUM_N];
        find_fn_and_fnp(paramb.basis_size_angular, paramb.rcinv_angular, used_d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          float gn12 = 0.0f;
          float gnp12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * dnnmb.c[c_index];
            gnp12 += fnp12[k] * dnnmb.c[c_index];
          }
          if (paramb.num_L == paramb.L_max) {
            accumulate_f12(n, paramb.n_max_angular + 1, used_d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else if (paramb.num_L == paramb.L_max + 1) {
            accumulate_f12_with_4body(
              n, paramb.n_max_angular + 1, used_d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else {
            accumulate_f12_with_5body(
              n, paramb.n_max_angular + 1, used_d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          }
        }
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NEP3::ZBL zbl,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s_pe = 0.0f;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int type1 = g_type[n1];
    float zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(zi, 0.23f);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      float zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(zj, 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
#ifdef USE_JESPER_HEA
      find_f_and_fp_zbl(type1, type2, zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
#else
      find_f_and_fp_zbl(zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
#endif
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
      s_pe += f * 0.5f;
    }
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
    g_pe[n1] += s_pe;
  }
}

void NEP3::find_force(
  Parameters& para,
  const float* parameters,
  float* parameters_grad,
  Dataset& dataset,
  bool calculate_q_scaler
) {
  nep_data.parameters.copy_from_host(parameters);
  update_potential(nep_data.parameters.data(), dnnmb);

  float rc2_radial = para.rc_radial * para.rc_radial;
  float rc2_angular = para.rc_angular * para.rc_angular;

  gpu_find_neighbor_list<<<dataset.Nc, 256>>>(
    dataset.N, dataset.Na.data(), dataset.Na_sum.data(), rc2_radial, rc2_angular,
    dataset.box.data(), dataset.box_original.data(), dataset.num_cell.data(), dataset.r.data(),
    dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2, nep_data.NN_radial.data(),
    nep_data.NL_radial.data(), nep_data.NN_angular.data(), nep_data.NL_angular.data(),
    nep_data.x12_radial.data(), nep_data.y12_radial.data(), nep_data.z12_radial.data(),
    nep_data.x12_angular.data(), nep_data.y12_angular.data(), nep_data.z12_angular.data());
  CUDA_CHECK_KERNEL

  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  find_descriptors_radial<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_radial.data(), nep_data.NL_radial.data(), paramb, dnnmb,
    dataset.type.data(), nep_data.x12_radial.data(), nep_data.y12_radial.data(),
    nep_data.z12_radial.data(), nep_data.descriptors.data());
  CUDA_CHECK_KERNEL

  find_descriptors_angular<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), paramb, dnnmb,
    dataset.type.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.descriptors.data(), nep_data.sum_fxyz.data());
  CUDA_CHECK_KERNEL

  if (calculate_q_scaler) {
    find_max_min<<<dnnmb.topology[0], 1024>>>(
      dataset.N, nep_data.descriptors.data(), para.q_scaler_gpu.data());
    CUDA_CHECK_KERNEL
  }

  int *dev_topology;
  cudaMalloc((void**)&dev_topology, sizeof(int) * dnnmb.n_layers);
  cudaMemcpy(dev_topology, dnnmb.topology, sizeof(int) * dnnmb.n_layers, cudaMemcpyHostToDevice);

  // ** APPLY DNN ** //
  apply_dnn<<<grid_size, block_size>>>(
    dataset.N,
    paramb,
    dnnmb,
    dev_topology,
    nep_data.descriptors.data(),
    para.q_scaler_gpu.data(),
    dataset.energy.data(),
    nep_data.Fp.data()
  );
  CUDA_CHECK_KERNEL

  zero_force<<<grid_size, block_size>>>(
    dataset.N, dataset.force.data(), dataset.force.data() + dataset.N,
    dataset.force.data() + dataset.N * 2);
  CUDA_CHECK_KERNEL

  // float *dev_coulomb_forces;
  // cudaMalloc((void**)&dev_coulomb_forces, sizeof(float) * dataset.N * 3);

  accumulate_radial_interactions<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_radial.data(), nep_data.NL_radial.data(), paramb, dnnmb, coulomb,
    dataset.type.data(), nep_data.x12_radial.data(), nep_data.y12_radial.data(),
    nep_data.z12_radial.data(), nep_data.Fp.data(), dataset.force.data(),
    dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2, dataset.virial.data(),
    // Added 2022 November 15, to enable us adding the Coulomb potential
    dataset.energy.data()
    // Added 2022 November 21, to enable us to calculate the effect of the Coulomb potential
    // dev_coulomb_forces
    // Added 2022 February 8, to enable us to use charges more effectively
    // charges
    );
  CUDA_CHECK_KERNEL

  // Copy back to host
  // float *coulomb_forces = new float[dataset.N * 3];
  // float *regular_forces = new float[dataset.N * 3];
  // cudaMemcpy(coulomb_forces, dev_coulomb_forces, sizeof(float) * dataset.N * 3, cudaMemcpyDeviceToHost);
  // cudaMemcpy(regular_forces, dataset.force.data(), sizeof(float) * dataset.N * 3, cudaMemcpyDeviceToHost);

  // cudaFree(dev_coulomb_forces);
  // delete regular_forces;

  // CUDA_CHECK_KERNEL

  // Skip this, so we can isolate Coulomb interactions
  find_force_angular<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), paramb, dnnmb,
    dataset.type.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.Fp.data(), nep_data.sum_fxyz.data(), dataset.force.data(),
    dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2, dataset.virial.data());
  CUDA_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, block_size>>>(
      dataset.N, zbl, nep_data.NN_angular.data(), nep_data.NL_angular.data(), dataset.type.data(),
      nep_data.x12_angular.data(), nep_data.y12_angular.data(), nep_data.z12_angular.data(),
      dataset.force.data(), dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2,
      dataset.virial.data(), dataset.energy.data());
    CUDA_CHECK_KERNEL
  }

  cudaFree(dev_topology);
  // cudaFree(charges);
}
