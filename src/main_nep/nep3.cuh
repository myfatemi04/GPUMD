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

#pragma once
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
class Parameters;
class Dataset;

struct NEP3_Data {
  GPU_Vector<int> NN_radial;  // radial neighbor number
  GPU_Vector<int> NL_radial;  // radial neighbor list
  GPU_Vector<int> NN_angular; // angular neighbor number
  GPU_Vector<int> NL_angular; // angular neighbor list
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;
  GPU_Vector<float> descriptors; // descriptors
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> parameters; // parameters to be optimized
};

class NEP3 : public Potential
{
public:
  struct ParaMB {
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int basis_size_radial = 0;  // for nep3
    int basis_size_angular = 0; // for nep3
    int n_max_radial = 0;       // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0;      // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;              // l = 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0; // for nep3
    int num_c_radial = 0; // for nep3
    int version = 2;      // 2 for NEP2 and 3 for NEP3
  };

  struct ANN {
    int dim = 0;          // dimension of the descriptor
    int num_neurons1 = 0; // number of neurons in the hidden layer
    int num_para = 0;     // number of parameters
    const float* w0;      // weight from the input layer to the hidden layer
    const float* b0;      // bias for the hidden layer
    const float* w1;      // weight from the hidden layer to the output layer
    const float* b1;      // bias for the output layer
    const float* c;
  };

  struct DNN {
    int num_para;
    int n_layers = 0;
    int* topology;         // number of neurons in nth layer. size is n_layers + 1. n=0 is input shape, n=n_layers is 1 (energy)
    
    int num_weights;
    int num_biases;
    const float* weights;
    const float* biases;
    const float* charges;  // charges between atoms
  };

  // Added by Michael Fatemi, 2022 November 12
  struct Coulomb {
    bool enabled = false;
    float alpha = 0.1f;
    // Should be set manually
    float epsilon = 0.0f;
    float charges[10];
  };

  struct ZBL {
    bool enabled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float atomic_numbers[10];
  };

  NEP3(
    char* input_dir,
    Parameters& para,
    int N,
    int N_times_max_NN_radial,
    int N_times_max_NN_angular,
    int version);
  void
  find_force(Parameters& para, const float* parameters, float* parameters_grad, Dataset& dataset, bool calculate_q_scaler);

private:
  ParaMB paramb;
  DNN dnnmb;
  NEP3_Data nep_data;
  ZBL zbl;
  Coulomb coulomb;
  void update_potential(const float* parameters, DNN& dnn);
};
