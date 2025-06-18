#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define N 800  // Tamaño de la matriz

void multiplicar_submatriz(double *A, double *B, double *C, int filas_locales, int columnas, int n_hilos) {
    #pragma omp parallel for num_threads(n_hilos)
    for (int i = 0; i < filas_locales; i++) {
        for (int j = 0; j < columnas; j++) {
            C[i * columnas + j] = 0.0;
            for (int k = 0; k < columnas; k++) {
                C[i * columnas + j] += A[i * columnas + k] * B[k * columnas + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *A, *B, *C, *A_local, *C_local;
    int filas_locales;
    double inicio, fin;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    filas_locales = N / size;

    A_local = (double *)malloc(filas_locales * N * sizeof(double));
    C_local = (double *)malloc(filas_locales * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        // Inicializar A y B
        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
    }

    // Medición de tiempo de ejecución
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizar antes de empezar
    inicio = MPI_Wtime();

    // Distribuir A a cada proceso
    MPI_Scatter(A, filas_locales * N, MPI_DOUBLE, A_local, filas_locales * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Difundir B a todos
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Obtener número de hilos desde variable de entorno
    int n_hilos = omp_get_max_threads();
    char *env = getenv("OMP_NUM_THREADS");
    if (env != NULL) n_hilos = atoi(env);

    // Multiplicación paralela
    multiplicar_submatriz(A_local, B, C_local, filas_locales, N, n_hilos);

    // Recolectar resultados
    MPI_Gather(C_local, filas_locales * N, MPI_DOUBLE, C, filas_locales * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    fin = MPI_Wtime();

    if (rank == 0) {
        printf("Tiempo total: %.4f segundos\n", fin - inicio);
    }

    // Liberar memoria
    if (rank == 0) {
        free(A);
        free(C);
    }
    free(A_local);
    free(B);
    free(C_local);

    MPI_Finalize();
    return 0;
}
