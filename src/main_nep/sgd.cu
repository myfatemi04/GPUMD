/*
    Copyright 2023 Michael Fatemi
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

#include <random>
#include <vector>

#include "fitness.cuh"
#include "parameters.cuh"
#include "sgd.cuh"
#include "utilities/error.cuh"
#include <chrono>
#include <cmath>

SGD::SGD(char* input_dir, Parameters& para, Fitness* fitness_function)
{
  maximum_generation = para.maximum_generation;
  number_of_variables = para.number_of_variables;
  fitness.resize(population_size * 6);
  nn_params.resize(para.number_of_variables);
  nn_params_grad.resize(para.number_of_variables);
  initialize_rng();
  compute(input_dir, para, fitness_function);
}

void SGD::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

void SGD::compute(char* input_dir, Parameters& para, Fitness* fitness_function)
{
  print_line_1();
  printf("Started training [SGD].\n");
  print_line_2();

  printf(
    "%-8s%-11s%-11s%-11s%-13s%-13s%-13s%-13s%-13s%-13s\n", "Step", "Total-Loss", "L1Reg-Loss",
    "L2Reg-Loss", "RMSE-E-Train", "RMSE-F-Train", "RMSE-V-Train", "RMSE-E-Test", "RMSE-F-Test",
    "RMSE-V-Test");

  init_nn(para);

  std::uniform_real_distribution<float> dis(0.0, 1.0);

  const float lr = 0.03;

  for (int n = 0; n < maximum_generation; ++n) {
    for (int i = 0; i < number_of_variables; i++) {
      nn_params_grad[i] = 0;
    }

    // Compute fitness function (and store grad)
    fitness_function->compute(n, para, nn_params.data(), fitness.data() + 3 * population_size, nn_params_grad.data());
    
    // printf("regularize")

    regularize(para);
    descend(lr);

    fitness_function->report_error(
      input_dir, para, n, fitness[0 + 0 * population_size], fitness[0 + 1 * population_size],
      fitness[0 + 2 * population_size], nn_params.data());
  }
}

void SGD::init_nn(Parameters& para)
{
  // std::normal_distribution<float> r1(0, 1);
  // for (int p = 0; p < population_size; ++p) {
  //   for (int v = 0; v < para.number_of_weights; ++v) {
  //     int pv = p * number_of_variables + v;
  //     nn_params[pv] = r1(rng);
  //   }
  //   for (int v = 0; v < para.number_of_biases; ++v) {
  //     int pv = p * number_of_variables + v + para.number_of_weights;
  //     nn_params[pv] = 0;
  //   }
  // }
  for (int v = 0; v < number_of_variables; ++v) {
    nn_params[v] = 0;
  }
  for (int i = 0; i < 4; i++) {
    nn_params[para.number_of_weights + para.number_of_biases - i - 1] = 3.10;
  }
  // nn_params[number_of_variables - 1] = 3.15;
  // nn_params[number_of_variables - 2] = 3.15;
  // nn_params[number_of_variables - 3] = 3.15;
  // nn_params[number_of_variables - 4] = 3.15;
}

// Actual gradient descent part
void SGD::descend(float lr)
{
  // Clip gradient to length 1
  float grad_l2 = 0;
  for (int i = 0; i < number_of_variables; i++) {
    grad_l2 += nn_params_grad[i] * nn_params_grad[i];
  }
  if (grad_l2 == 0) {
    return;
  }
  float scale_by;
  if (grad_l2 <= 1) {
    scale_by = 1;
  } else {
    scale_by = 1 / sqrt(grad_l2);
  }
  printf("Running gradient descent: grad_l2 = %f\n", grad_l2);
  for (int i = 0; i < number_of_variables; i++) {
    nn_params[i] -= nn_params_grad[i] * lr * scale_by;
  }
}

void SGD::regularize(Parameters& para)
{
  for (int p = 0; p < population_size; ++p) {
    // Update fitness values and add weight decay
    float cost_L1 = 0.0f, cost_L2 = 0.0f;
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      cost_L1 += std::abs(nn_params[pv]);
      cost_L2 += nn_params[pv] * nn_params[pv];
    }
    cost_L1 *= para.lambda_1 / number_of_variables;
    cost_L2 = para.lambda_2 * sqrt(cost_L2 / number_of_variables);
    fitness[p] = cost_L1 + cost_L2 + fitness[p + 3 * population_size] +
                 fitness[p + 4 * population_size] + fitness[p + 5 * population_size];
    fitness[p + 1 * population_size] = cost_L1;
    fitness[p + 2 * population_size] = cost_L2;

    // Don't apply weight decay to biases
    // for (int i = 0; i < para.number_of_weights; i++) {
    //   // loss += param^2 * lambda_2 + param * lambda_1
    //   // dloss/dparam += param * lambda_2 + lambda_1
    //   nn_params_grad[i] += 2 * nn_params[i] * para.lambda_2;
    //   nn_params_grad[i] += 1 * para.lambda_1;
    // }
  }
}
