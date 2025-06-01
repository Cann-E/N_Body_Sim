#include "common.h"
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256


int blks;
int gridDimX;
int gridDimY;
double cellSize;

// Device buffers
int* cell_ids = nullptr;
int* particle_ids = nullptr;
int* cell_start = nullptr;
int* cell_end = nullptr;
particle_t* sorted = nullptr;

// unchanged
__device__ void apply_force_gpu(particle_t& p, const particle_t& n) {
    double dx = n.x - p.x;
    double dy = n.y - p.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    p.ax += coef * dx;
    p.ay += coef * dy;
}

// assign particles to cells
__global__ void assign_cells(particle_t* parts, int* cell_ids, int* particle_ids, int num_parts, double cellSize, int gridDimX) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int cx = (int)(parts[tid].x / cellSize);
    int cy = (int)(parts[tid].y / cellSize);
    cx = max(0, min(gridDimX - 1, cx));
    cy = max(0, min(gridDimX - 1, cy));
    int cell = cy * gridDimX + cx;

    cell_ids[tid] = cell;
    particle_ids[tid] = tid;
}

// build cell start/end maps
__global__ void build_cell_bounds(int* cell_ids, int* cell_start, int* cell_end, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int curr = cell_ids[tid];
    if (tid == 0) {
        cell_start[curr] = 0;
    } else {
        int prev = cell_ids[tid - 1];
        if (curr != prev) {
            cell_start[curr] = tid;
            cell_end[prev] = tid;
        }
    }
    if (tid == num_parts - 1) {
        cell_end[curr] = num_parts;
    }
}

// reorder particles
__global__ void reorder_particles(particle_t* src, particle_t* dst, int* map, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    dst[tid] = src[map[tid]];
}

// compute forces in 3x3 neighbor grid
__global__ void compute_forces_gpu_grid(particle_t* parts, particle_t* sorted, int* particle_ids, int* cell_start, int* cell_end, int num_parts, double cellSize, int gridDimX) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t p = sorted[tid];
    p.ax = 0;
    p.ay = 0;

    int cx = (int)(p.x / cellSize);
    int cy = (int)(p.y / cellSize);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int ncx = cx + dx;
            int ncy = cy + dy;
            if (ncx < 0 || ncx >= gridDimX || ncy < 0 || ncy >= gridDimX) continue;

            int ncell = ncy * gridDimX + ncx;
            int start = cell_start[ncell];
            int end = cell_end[ncell];
            if (start == -1 || end == -1) continue;

            for (int i = start; i < end; i++) {
                apply_force_gpu(p, sorted[i]);
            }
        }
    }

    int idx = particle_ids[tid];
    parts[idx].ax = p.ax;
    parts[idx].ay = p.ay;
}

// unchanged
__global__ void move_gpu(particle_t* parts, int n, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t* p = &parts[tid];
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    if (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    if (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

// set grid info
void init_simulation(particle_t* parts, int n, double size) {
    blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    cellSize = cutoff;
    gridDimX = (int)(size / cellSize) + 1;
    gridDimY = gridDimX;
}

// simulate one step
void simulate_one_step(particle_t* parts, int n, double size) {
    int num_cells = gridDimX * gridDimY;

    if (!cell_ids) {
        cudaMalloc(&cell_ids, n * sizeof(int));
        cudaMalloc(&particle_ids, n * sizeof(int));
        cudaMalloc(&sorted, n * sizeof(particle_t));
        cudaMalloc(&cell_start, num_cells * sizeof(int));
        cudaMalloc(&cell_end, num_cells * sizeof(int));
    }

    assign_cells<<<blks, NUM_THREADS>>>(parts, cell_ids, particle_ids, n, cellSize, gridDimX);

    thrust::device_ptr<int> d_cell_ids(cell_ids);
    thrust::device_ptr<int> d_particle_ids(particle_ids);
    thrust::sort_by_key(d_cell_ids, d_cell_ids + n, d_particle_ids);

    reorder_particles<<<blks, NUM_THREADS>>>(parts, sorted, particle_ids, n);

    cudaMemset(cell_start, -1, num_cells * sizeof(int));
    cudaMemset(cell_end, -1, num_cells * sizeof(int));

    build_cell_bounds<<<blks, NUM_THREADS>>>(cell_ids, cell_start, cell_end, n);

    compute_forces_gpu_grid<<<blks, NUM_THREADS>>>(parts, sorted, particle_ids, cell_start, cell_end, n, cellSize, gridDimX);

    move_gpu<<<blks, NUM_THREADS>>>(parts, n, size);
}
