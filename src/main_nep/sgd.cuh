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

#pragma once
#include <random>
#include <vector>
class Fitness;
class Fitness;

class SGD
{
public:
  SGD(char*, Parameters&, Fitness*);

protected:
  std::mt19937 rng;
  int maximum_generation = 10000;
  int number_of_variables = 10;
  const int population_size = 1;
  std::vector<float> nn_params;
  std::vector<float> nn_params_grad;
  std::vector<float> fitness;
  void init_nn(Parameters&);
  void initialize_rng();
  void compute(char*, Parameters&, Fitness*);
  void regularize(Parameters&);
  void descend(float lr);
};
