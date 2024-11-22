#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

int main(int argc, char** argv) {
    int rank, size;
    int t, n, m, D1, D2;
    int* At = NULL;  // At     matrix
    int* K = NULL;   // kernal matrix
    int D1_half, D2_half;
    int* recvcounts = NULL;
    int* displs = NULL;
    int local_row_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // read the data
    if (rank == 0) {
        char file_name[50];
        scanf("%s", file_name);
        FILE *file = fopen(file_name, "r");

        fscanf(file, "%d", &t);
        fscanf(file, "%d %d", &n, &m);

        At = (int*)malloc(n * m * sizeof(int));
        for (int i = 0; i< m; i++) {
            for (int j = 0; j < n; j++) {
                fscanf(file, "%d", At + (i * n + j));
            }
        }

        fscanf(file, "%d %d", &D1, &D2);
        K = (int*)malloc(D1* D2 * sizeof(int));
        for (int i = 0; i< D1 * D2; i++) {
            fscanf(file, "%d", K + i);
        }
        fclose(file);
    }

    //Bcast the data
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    D1_half = (D1 - 1) / 2;
    D2_half = (D2 - 1) / 2;

    // broad cast K before
    if (rank != 0) {
        // make sure there is enough memory in other threads
        K = (int*)malloc(D1* D2* sizeof(int));
    }
    MPI_Bcast(K, D1 * D2, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = m / size;
    int extra_rows = m % size;

    recvcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int rows = (i < extra_rows) ? rows_per_proc + 1 : rows_per_proc;
        recvcounts[i] = rows * n;
        displs[i] = offset;
        offset += recvcounts[i];
    }

    local_row_count = recvcounts[rank] / n;

    int* local_At = NULL;
    int* local_An = NULL;
    local_At = (int*)malloc((local_row_count + 2 * D1_half) * n * sizeof(int));
    local_An = (int*)malloc(local_row_count * n * sizeof(int)); //每個rank需要處理的元素量不同

    // 初始化 At，仅在 0 号进程有完整的 At
    if (rank != 0) {
        At = (int*)malloc((m + 2 * D2_half) * n * sizeof(int));
    } else {
        // 扩展 At，用于处理边界情况（环绕地球）
        int* temp_At = (int*)malloc((m + 2 * D2_half) * n * sizeof(int));
        // 将 At 复制到 temp_At 的中间部分
        memcpy(temp_At + D2_half * n, At, m * n * sizeof(int));
        // *处理环绕，复制上下边界
        for (int i = 0; i < D2_half; ++i) {
            memcpy(temp_At + i * n, At + (m - D2_half + i) * n, n * sizeof(int)); // 上边界
            memcpy(temp_At + (m + D2_half + i) * n, At + i * n, n * sizeof(int)); // 下边界
        }
        free(At);
        At = temp_At;
    }

    // 广播扩展后的 At v
    MPI_Bcast(At, (m + 2 * D2_half) * n, MPI_INT, 0, MPI_COMM_WORLD);

    int start_row = displs[rank] / n;
    // main loop
    for (int time = 0; time < t; ++time) {
        // 对于每个时间步，都需要更新 At
        // 每个进程提取自己需要的数据，包括上下 D_half 行的重叠区域
        int global_start = start_row;
        int local_data_start = global_start;
        memcpy(local_At, At + local_data_start * n, (local_row_count + 2 * D1_half) * n * sizeof(int));

        // 计算本地的 An
        for (int i = D1_half; i < local_row_count + D1_half; ++i) {
            for (int j = 0; j < n; ++j) {
                int sum = 0;
                for (int di = -D1_half; di <= D1_half; ++di) { // cool!
                    int ii = i + di;
                    for (int dj = -D2_half; dj <= D2_half; ++dj) {
                        int jj = mod(j + dj, n);
                        int ki = D1_half + di;
                        int kj = D2_half + dj;
                        sum += K[ki * D2 + kj] * local_At[ii * n + jj];
                    }
                }
                local_An[(i - D1_half) * n + j] = sum / (D1 * D2);
            }
        }

        // 所有进程将计算结果收集到 At 中
        MPI_Allgatherv(local_An, local_row_count * n, MPI_INT,
                       At + D1_half * n, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);

        // 处理上下边界的环绕
        for (int i = 0; i < D2_half; ++i) {
            // 上边界
            memcpy(At + i * n, At + (m + i) * n, n * sizeof(int));
            // 下边界
            memcpy(At + (m + D2_half + i) * n, At + (D2_half + i) * n, n * sizeof(int));
        }
    }

    // 输出结果
    if (rank == 0) {
        for (int i = D2_half; i < m + D2_half; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%d ", At[i * n + j]);
            }
        }
    }

    // 释放内存
    free(At);
    free(K);
    free(local_At);
    free(local_An);
    free(recvcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}