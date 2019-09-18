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
Green-Kubo Modal Analysis (GKMA)
- Currently only supports output of modal heat flux
 -> Green-Kubo integrals must be post-processed

GPUMD Contributing author: Alexander Gabourie (Stanford University)

Some code here and supporting code in 'potential.cu' is based on the LAMMPS
implementation provided by the Henry group at MIT. This code can be found:
https://drive.google.com/open?id=1IHJ7x-bLZISX3I090dW_Y_y-Mqkn07zg
------------------------------------------------------------------------------*/

#include "gkma.cuh"
#include "atom.cuh"
#include <fstream>
#include <string>
#include <iostream>

#define BLOCK_SIZE 128


static __global__ void gpu_reset_gkma_data
(
        int num_elements, real* xdot, real* jm
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_elements)
    {
        xdot[n] = ZERO;
        jm[n] = ZERO;
    }
}

static __global__ void gpu_average_jm
(
        int num_elements, int samples_per_output, real* jm
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_elements)
    {
        jm[n]/=(float)samples_per_output;
    }
}

void GKMA::preprocess(char *input_dir, Atom *atom)
{
    num_modes = last_mode-first_mode+1;
    samples_per_output = output_interval/sample_interval;
    num_bins = num_modes/bin_size;

    strcpy(gkma_file_position, input_dir);
    strcat(gkma_file_position, "/heatmode.out");

    // initialize eigenvector data structures
    strcpy(eig_file_position, input_dir);
    strcat(eig_file_position, "/eigenvector.eig");
    std::ifstream eigfile;
    eigfile.open(eig_file_position);
    if (!eigfile)
    {
        print_error("Cannot open eigenvector.eig file.\n");
    }

    int N = atom->N;
    CHECK(cudaMallocManaged(&eig, sizeof(real) * N * num_modes * 3));

    // Following code snippet is heavily based on MIT LAMMPS code
    std::string val;
    double doubleval;

    for (int i=0; i<=N+3 ; i++){
        getline(eigfile,val);
    }
    for (int i=0; i<first_mode-1; i++){
      for (int j=0; j<N+2; j++) getline(eigfile,val);
    }
    for (int j=0; j<num_modes; j++){
        getline(eigfile,val);
        getline(eigfile,val);
        for (int i=0; i<N; i++){
            eigfile >> doubleval;
            eig[i + 3*N*j] = doubleval;
            eigfile >> doubleval;
            eig[i + (1 + 3*j)*N] = doubleval;
            eigfile >> doubleval;
            eig[i + (2 + 3*j)*N] = doubleval;
        }
        getline(eigfile,val);
    }
    eigfile.close();
    //end snippet

    // Allocate modal variables
    CHECK(cudaMallocManaged(&xdot, sizeof(real) * num_modes * 3));
    CHECK(cudaMallocManaged(&jm, sizeof(real) * num_modes * 3));

}

void GKMA::process(int step)
{
    if (!compute) return;
    if (!((step+1) % output_interval == 0)) return;

    int num_elements = num_modes*3;
    gpu_average_jm<<<(num_elements-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
    (
            num_elements, samples_per_output, jm
    );
    CUDA_CHECK_KERNEL

    // TODO make into a GPU function
    real *bin_out; // bins of heat current modes for output
    ZEROS(bin_out, real, 3*num_bins);
    for (int i = 0; i < num_bins; i++)
    {
        for (int j = 0; j < bin_size; j++)
        {
            bin_out[i] += jm[j + i*bin_size];
            bin_out[i + num_bins] += jm[j + i*bin_size + num_modes];
            bin_out[i + 2*num_bins] += jm[j + i*bin_size + 2*num_modes];
        }
    }

    FILE *fid = fopen(gkma_file_position, "a");
    for (int i = 0; i < num_bins; i++)
    {
        fprintf(fid, "%25.15e %25.15e %25.15e\n",
                bin_out[i], bin_out[i+num_bins], bin_out[i+2*num_bins]);
    }
    fflush(fid);
    fclose(fid);
    MY_FREE(bin_out);

    gpu_reset_gkma_data<<<(num_elements-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
    (
            num_elements, xdot, jm
    );
    CUDA_CHECK_KERNEL
}

void GKMA::postprocess()
{
    if (!compute) return;
    CHECK(cudaFree(eig));
    CHECK(cudaFree(xdot));
    CHECK(cudaFree(jm));
}


