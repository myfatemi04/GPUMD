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
#include "mc.cuh"
#include "utilities/error.cuh"
#include <chrono>
#include <cmath>

MetropolisMonteCarlo::MetropolisMonteCarlo(char* input_dir, Parameters& para, Fitness* fitness_function)
{
  maximum_generation = para.maximum_generation;
  number_of_variables = para.number_of_variables;
  population_size = para.population_size;
  // eta_sigma = (3.0f + std::log(number_of_variables * 1.0f)) /
  //             (5.0f * sqrt(number_of_variables * 1.0f)) / 2.0f;
  fitness.resize(population_size * 6);
  fitness_copy.resize(population_size * 6);
  index.resize(population_size);
  population.resize(population_size * number_of_variables);
  population_copy.resize(population_size * number_of_variables);
  initialize_rng();
  compute(input_dir, para, fitness_function);
}

void MetropolisMonteCarlo::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

void MetropolisMonteCarlo::copy_population() {
  for (int i = 0; i < population.size(); i++) {
    population_copy[i] = population[i];
  }
}

// Outputs a random step to `step`, normalized to magnitude of `step_size`
void MetropolisMonteCarlo::random_step(int dim, float step_size, float *step) {
  std::normal_distribution<float> r1(0, 1);
  float l2 = 0;
  for (int j = 0; j < dim; j++) {
    step[j] = r1(rng);
    l2 += step[j] * step[j];
  }
  l2 = sqrt(l2);
  for (int j = 0; j < dim; j++) {
    step[j] = step[j] / l2;
  }
}

void MetropolisMonteCarlo::compute(char* input_dir, Parameters& para, Fitness* fitness_function)
{
  print_line_1();
  printf("Started training [MetropolisMonteCarlo].\n");
  print_line_2();

  printf(
    "%-8s%-11s%-11s%-11s%-13s%-13s%-13s%-13s%-13s%-13s\n", "Step", "Total-Loss", "L1Reg-Loss",
    "L2Reg-Loss", "RMSE-E-Train", "RMSE-F-Train", "RMSE-V-Train", "RMSE-E-Test", "RMSE-F-Test",
    "RMSE-V-Test");

  // Create population in beginning only
  create_population(para);

  std::uniform_real_distribution<float> dis(0.0, 1.0);

  for (int n = 0; n < maximum_generation; ++n) {
    // Compute fitness function
    fitness_function->compute(n, para, population.data(), fitness.data() + 3 * population_size);
    regularize(para, false);

    copy_population();

    // Compare change and choose whether to move in that direction
    float *step = new float[number_of_variables];
    float step_size = 0.1;
    for (int i = 0; i < population_size; i++) {
      // Outputs to `step`
      random_step(number_of_variables, step_size, step);
      
      for (int j = 0; j < number_of_variables; j++) {
        population_copy[i * number_of_variables + j] += step[j];
      }
    }

    // Recalculate fitness
    fitness_function->compute(n, para, population.data(), fitness_copy.data() + 3 * population_size);
    regularize(para, true);

    float temperature = 0.05;

    // Compare fitness: fitness[0:population_size] is the overall fitness
    for (int i = 0; i < population_size; i++) {
      float delta_fitness = fitness_copy[i] - fitness[i];
      // If delta_fitness > 0 (fitness improved), dis(rng) is ALWAYS less.
      // If delta_fitness < 0 (fitness decreased), dis(rng) MAY BE less.
      if (dis(rng) <= exp(delta_fitness / temperature)) {
        // The change was accepted.
        for (int j = 0; j < number_of_variables; j++) {
          population[i * number_of_variables + j] = population_copy[i * number_of_variables + j];
        }
        for (int k = 0; k < 6; k++) {
          fitness[i * 6 + k] = fitness_copy[i * 6 + k];
        }
      }
    }

    fitness_function->report_error(
      input_dir, para, n, fitness[0 + 0 * population_size], fitness[0 + 1 * population_size],
      fitness[0 + 2 * population_size], population.data());
  }
}

void MetropolisMonteCarlo::create_population(Parameters& para)
{
  std::normal_distribution<float> r1(0, 1);
  for (int p = 0; p < population_size; ++p) {
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      // s[pv] = r1(rng);
      population[pv] = r1(rng); // s[pv] + mu[v];
      // avoid zero
      if (v >= para.number_of_variables_dnn) {
        if (population[pv] > 0) {
          population[pv] += 0.1f;
        } else {
          population[pv] -= 0.1f;
        }
      }
    }
  }
}

void MetropolisMonteCarlo::regularize(Parameters& para, bool use_copy)
{
  std::vector<float>& fitness_ = use_copy ? fitness_copy : fitness;
  for (int p = 0; p < population_size; ++p) {
    float cost_L1 = 0.0f, cost_L2 = 0.0f;
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      cost_L1 += std::abs(population[pv]);
      cost_L2 += population[pv] * population[pv];
    }
    cost_L1 *= para.lambda_1 / number_of_variables;
    cost_L2 = para.lambda_2 * sqrt(cost_L2 / number_of_variables);
    fitness_[p] = cost_L1 + cost_L2 + fitness_[p + 3 * population_size] +
                 fitness_[p + 4 * population_size] + fitness_[p + 5 * population_size];
    fitness_[p + 1 * population_size] = cost_L1;
    fitness_[p + 2 * population_size] = cost_L2;
  }
}
