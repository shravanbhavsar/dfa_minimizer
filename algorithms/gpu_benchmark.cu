// gpu_benchmark.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel: mark mask[i]=1 if state i has a transition into `splitter`
__global__ void compute_target_partitions(
    const int *T,    // flattened n×k transition table
    const int *part, // partition IDs array, length n
    int n,           // number of states
    int k,           // alphabet size
    int splitter,    // partition ID to test against
    int *mask        // output mask, length n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int m = 0;
    for (int j = 0; j < k; ++j) {
        int tgt = T[idx*k + j];
        if (part[tgt] == splitter) { m = 1; break; }
    }
    mask[idx] = m;
}

// CUDA kernel: for each i with part[i]==q && mask[i]==1, assign part[i]=new_q
__global__ void split_partition(
    int *part,
    const int *mask,
    int n,
    int q,
    int new_q,
    int *flag      // device flag: set to 1 if any split occurred
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (part[idx] == q && mask[idx]) {
        part[idx] = new_q;
        *flag = 1;
    }
}

// Host function: GPU-based partition refinement
void gpu_minimize(int *h_T, int *h_part, int n, int k) {
    int *d_T, *d_part, *d_mask, *d_flag;
    size_t size_T = n * k * sizeof(int);
    size_t size_n = n * sizeof(int);

    cudaMalloc(&d_T, size_T);
    cudaMalloc(&d_part, size_n);
    cudaMalloc(&d_mask, size_n);
    cudaMalloc(&d_flag, sizeof(int));

    cudaMemcpy(d_T, h_T, size_T, cudaMemcpyHostToDevice);
    cudaMemcpy(d_part, h_part, size_n, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (n + threads - 1) / threads;

    // Worklist of partition IDs
    int *worklist = (int*)malloc((n+2)*sizeof(int));
    int wl_size = 0;
    worklist[wl_size++] = 0;  // finals
    worklist[wl_size++] = 1;  // non-finals
    int max_q = 1;

    while (wl_size > 0) {
        int splitter = worklist[--wl_size];

        // Compute mask
        compute_target_partitions<<<blocks,threads>>>(d_T, d_part, n, k, splitter, d_mask);
        cudaDeviceSynchronize();

        // Try splitting each partition q = 0..max_q
        for (int q = 0; q <= max_q; ++q) {
            int zero = 0;
            cudaMemcpy(d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);

            int new_q = max_q + 1;
            split_partition<<<blocks,threads>>>(d_part, d_mask, n, q, new_q, d_flag);
            cudaDeviceSynchronize();

            int changed;
            cudaMemcpy(&changed, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (changed) {
                // Pull part back to decide smaller piece
                int *temp = (int*)malloc(size_n);
                cudaMemcpy(temp, d_part, size_n, cudaMemcpyDeviceToHost);
                int cnt1=0, cnt2=0;
                for (int i = 0; i < n; ++i) {
                    if (temp[i] == new_q) ++cnt1;
                    else if (temp[i] == q) ++cnt2;
                }
                free(temp);
                worklist[wl_size++] = (cnt1 < cnt2 ? new_q : q);
                max_q = new_q;
            }
        }
    }

    cudaMemcpy(h_part, d_part, size_n, cudaMemcpyDeviceToHost);

    free(worklist);
    cudaFree(d_T);
    cudaFree(d_part);
    cudaFree(d_mask);
    cudaFree(d_flag);
}

int main() {
    // Benchmark parameters
    int sizes[] = {100, 500, 1000};
    int numSizes = sizeof(sizes)/sizeof(sizes[0]);
    int trials = 3;
    int k = 2;  // binary alphabet
    srand((unsigned)time(NULL));

    for (int si = 0; si < numSizes; ++si) {
        int n = sizes[si];
        float total_ms = 0.0f;

        for (int t = 0; t < trials; ++t) {
            // Allocate host arrays
            int *h_T    = (int*)malloc(n * k * sizeof(int));
            int *h_part = (int*)malloc(n * sizeof(int));

            // 1) Random transitions
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < k; ++j) {
                    h_T[i*k + j] = rand() % n;
                }
            }
            // 2) Random final‐state subset
            int numF = rand() % n + 1;
            int *idxs = (int*)malloc(n * sizeof(int));
            for (int i = 0; i < n; ++i) idxs[i] = i;
            for (int i = 0; i < numF; ++i) {
                int r = i + rand() % (n - i);
                int tmp = idxs[i]; idxs[i]=idxs[r]; idxs[r]=tmp;
            }
            // initial partition: 0=final, 1=non-final
            for (int i = 0; i < n; ++i) h_part[i] = 1;
            for (int i = 0; i < numF; ++i) h_part[idxs[i]] = 0;
            free(idxs);

            // Measure GPU minimization time
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            gpu_minimize(h_T, h_part, n, k);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            free(h_T);
            free(h_part);
        }

        printf("n = %4d : GPU avg time = %8.3f ms\n", n, total_ms / trials);
    }
    return 0;
}
