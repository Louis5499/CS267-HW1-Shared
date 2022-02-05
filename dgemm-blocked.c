#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define min(a, b) (((a) < (b)) ? (a) : (b))

static inline __attribute__((optimize("unroll-loops"))) void microkernel_8by8(double* A, double* B, double* C) {
    int MICROKERNEL_SIZE = 8;
    __m512d cij, cij1, cij2, cij3, cij4, cij5, cij6, cij7;
    __m512d aik;
    asm volatile (
        "vmovapd 0(%[C]), %[cij]\n\t"
        "vmovapd 64(%[C]), %[cij1]\n\t"
        "vmovapd 128(%[C]), %[cij2]\n\t"
        "vmovapd 192(%[C]), %[cij3]\n\t"
        "vmovapd 256(%[C]), %[cij4]\n\t"
        "vmovapd 320(%[C]), %[cij5]\n\t"
        "vmovapd 384(%[C]), %[cij6]\n\t"
        "vmovapd 448(%[C]), %[cij7]\n\t"
        : [cij] "+v" (cij),
        [cij1] "+v" (cij1),
        [cij2] "+v" (cij2),
        [cij3] "+v" (cij3),
        [cij4] "+v" (cij4),
        [cij5] "+v" (cij5),
        [cij6] "+v" (cij6),
        [cij7] "+v" (cij7)
        : [C] "r" (C)
        : "memory"
    );
    for (int k=0; k<MICROKERNEL_SIZE; k+=1) {
        double* Aik = A + (k * MICROKERNEL_SIZE);
        double* Bkj = B + k;
        asm volatile (
            "vmovapd (%[Aik]), %[aik]\n\t"
            "vfmadd231pd 0(%[Bkj])%{1to8%}, %[aik], %[cij]\n\t"
            "vfmadd231pd 64(%[Bkj])%{1to8%}, %[aik], %[cij1]\n\t"
            "vfmadd231pd 128(%[Bkj])%{1to8%}, %[aik], %[cij2]\n\t"
            "vfmadd231pd 192(%[Bkj])%{1to8%}, %[aik], %[cij3]\n\t"
            "vfmadd231pd 256(%[Bkj])%{1to8%}, %[aik], %[cij4]\n\t"
            "vfmadd231pd 320(%[Bkj])%{1to8%}, %[aik], %[cij5]\n\t"
            "vfmadd231pd 384(%[Bkj])%{1to8%}, %[aik], %[cij6]\n\t"
            "vfmadd231pd 448(%[Bkj])%{1to8%}, %[aik], %[cij7]\n\t"
            : [aik] "+v" (aik),
            [cij] "+v" (cij),
            [cij6] "+v" (cij6),
            [cij5] "+v" (cij5),
            [cij4] "+v" (cij4),
            [cij2] "+v" (cij2),
            [cij3] "+v" (cij3),
            [cij7] "+v" (cij7),
            [cij1] "+v" (cij1)
            : [Aik] "r" (Aik),
            [Bkj] "r" (Bkj)
            : "memory"
        );
    }
    asm volatile (
        "vmovapd %[cij], 0(%[C])\n\t"
        "vmovapd %[cij1], 64(%[C])\n\t"
        "vmovapd %[cij2], 128(%[C])\n\t"
        "vmovapd %[cij3], 192(%[C])\n\t"
        "vmovapd %[cij4], 256(%[C])\n\t"
        "vmovapd %[cij5], 320(%[C])\n\t"
        "vmovapd %[cij6], 384(%[C])\n\t"
        "vmovapd %[cij7], 448(%[C])\n\t"
        : [cij3] "+v" (cij3),
        [cij4] "+v" (cij4),
        [cij6] "+v" (cij6),
        [cij2] "+v" (cij2),
        [cij5] "+v" (cij5),
        [cij1] "+v" (cij1),
        [cij] "+v" (cij),
        [cij7] "+v" (cij7)
        : [C] "r" (C)
        : "memory"
    );
}

static inline __attribute__((optimize("unroll-loops"))) void do_block(int n, int M, int N, int K, double* A, double* B, double* C) {
    int MICROKERNEL_SIZE = 8;
    double* block_A = (double *)_mm_malloc(MICROKERNEL_SIZE * MICROKERNEL_SIZE * sizeof(double), 64);
    double* block_B = (double *)_mm_malloc(MICROKERNEL_SIZE * MICROKERNEL_SIZE * sizeof(double), 64);
    double* block_C = (double *)_mm_malloc(MICROKERNEL_SIZE * MICROKERNEL_SIZE * sizeof(double), 64);

    // For each block-row of A
    for (int i = 0; i < M; i += MICROKERNEL_SIZE) {
        for (int j = 0; j < N; j += MICROKERNEL_SIZE) {
            for (int k = 0; k < K; k += MICROKERNEL_SIZE) {

                // Initialize blocks starting at point (i,j,k)
                for (int y = 0; y < MICROKERNEL_SIZE; y++) {
                    for (int x = 0; x < MICROKERNEL_SIZE; x++) {
                        block_A[x + y * MICROKERNEL_SIZE] = A[i + k * n + x + y * n];
                        // printf("Block A at %d gets A at %d\n", x + y * MICROKERNEL_SIZE, k + j * n + x + y * n);
                        // fflush(stdout);
                        block_B[x + y * MICROKERNEL_SIZE] = B[k + j * n + x + y * n];
                        block_C[x + y * MICROKERNEL_SIZE] = C[i + j * n + x + y * n];
                    }
                }

                    // printf("block A before microkernel\n");
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[0], block_A[8], block_A[16], block_A[24], block_A[32], block_A[40], block_A[48], block_A[56]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[1], block_A[9], block_A[17], block_A[25], block_A[33], block_A[41], block_A[49], block_A[57]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[2], block_A[10], block_A[18], block_A[26], block_A[34], block_A[42], block_A[50], block_A[58]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[3], block_A[11], block_A[19], block_A[27], block_A[35], block_A[43], block_A[51], block_A[59]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[4], block_A[12], block_A[20], block_A[28], block_A[36], block_A[44], block_A[52], block_A[60]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[5], block_A[13], block_A[21], block_A[29], block_A[37], block_A[45], block_A[53], block_A[61]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[6], block_A[14], block_A[22], block_A[30], block_A[38], block_A[46], block_A[54], block_A[62]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_A[7], block_A[15], block_A[23], block_A[31], block_A[39], block_A[47], block_A[55], block_A[63]);
                    // printf("block B before microkernel\n");
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[0], block_B[8], block_B[16], block_B[24], block_B[32], block_B[40], block_B[48], block_B[56]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[1], block_B[9], block_B[17], block_B[25], block_B[33], block_B[41], block_B[49], block_B[57]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[2], block_B[10], block_B[18], block_B[26], block_B[34], block_B[42], block_B[50], block_B[58]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[3], block_B[11], block_B[19], block_B[27], block_B[35], block_B[43], block_B[51], block_B[59]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[4], block_B[12], block_B[20], block_B[28], block_B[36], block_B[44], block_B[52], block_B[60]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[5], block_B[13], block_B[21], block_B[29], block_B[37], block_B[45], block_B[53], block_B[61]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[6], block_B[14], block_B[22], block_B[30], block_B[38], block_B[46], block_B[54], block_B[62]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_B[7], block_B[15], block_B[23], block_B[31], block_B[39], block_B[47], block_B[55], block_B[63]);
                    // printf("block C before microkernel\n");
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[0], block_C[8], block_C[16], block_C[24], block_C[32], block_C[40], block_C[48], block_C[56]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[1], block_C[9], block_C[17], block_C[25], block_C[33], block_C[41], block_C[49], block_C[57]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[2], block_C[10], block_C[18], block_C[26], block_C[34], block_C[42], block_C[50], block_C[58]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[3], block_C[11], block_C[19], block_C[27], block_C[35], block_C[43], block_C[51], block_C[59]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[4], block_C[12], block_C[20], block_C[28], block_C[36], block_C[44], block_C[52], block_C[60]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[5], block_C[13], block_C[21], block_C[29], block_C[37], block_C[45], block_C[53], block_C[61]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[6], block_C[14], block_C[22], block_C[30], block_C[38], block_C[46], block_C[54], block_C[62]);
                    // printf("%f %f %f %f %f %f %f %f\n", block_C[7], block_C[15], block_C[23], block_C[31], block_C[39], block_C[47], block_C[55], block_C[63]);                 
                    // fflush(stdout);
                            
                // Perform individual block dgemm
                microkernel_8by8(block_A, block_B, block_C);

                // printf("block C after microkernel\n");
                // printf("%f %f %f %f %f %f %f %f\n", block_C[0], block_C[8], block_C[16], block_C[24], block_C[32], block_C[40], block_C[48], block_C[56]);
                // printf("%f %f %f %f %f %f %f %f\n", block_C[1], block_C[9], block_C[17], block_C[25], block_C[33], block_C[41], block_C[49], block_C[57]);
                // printf("%f %f %f %f %f %f %f %f\n", block_C[2], block_C[10], block_C[18], block_C[26], block_C[34], block_C[42], block_C[50], block_C[58]);
                // printf("%f %f %f %f %f %f %f %f\n", block_C[3], block_C[11], block_C[19], block_C[27], block_C[35], block_C[43], block_C[51], block_C[59]);
                // printf("%f %f %f %f %f %f %f %f\n", block_C[4], block_C[12], block_C[20], block_C[28], block_C[36], block_C[44], block_C[52], block_C[60]);
                // printf("%f %f %f %f %f %f %f %f\n", block_C[5], block_C[13], block_C[21], block_C[29], block_C[37], block_C[45], block_C[53], block_C[61]);
                // printf("%f %f %f %f %f %f %f %f\n", block_C[6], block_C[14], block_C[22], block_C[30], block_C[38], block_C[46], block_C[54], block_C[62]);
                // printf("%f %f %f %f %f %f %f %f\n", block_C[7], block_C[15], block_C[23], block_C[31], block_C[39], block_C[47], block_C[55], block_C[63]);

                // Read our block c values into original c.
                for (int y = 0; y < MICROKERNEL_SIZE; y++) {
                    for (int x = 0; x < MICROKERNEL_SIZE; x++) {
                        C[i + j * n + x + y * n] = block_C[x + y * MICROKERNEL_SIZE];
                    }
                }
            }
        }
    }
    free(block_A);
    free(block_B);
    free(block_C);
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    int n;
    if (lda % 16 == 0) {
        n = lda;
    } else {
        n = lda + (16 - (lda%16));
    }

    // TODO: Allocate smaller block inside do_block_ins
    double *A_trans = (double *) _mm_malloc(n * n * sizeof(double), 64);
    double *B_cpy = (double *) _mm_malloc(n * n * sizeof(double), 64);
    double *C_cpy = (double *) _mm_malloc(n * n * sizeof(double), 64);
    memset(C_cpy, 0, n * n * sizeof(double));

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i >= lda || j >= lda) {
                A_trans[i + n * j] = 0;
                B_cpy[i + n * j] = 0;
            } else {
                A_trans[i + n * j] = A[i + lda * j];
                B_cpy[i + n * j] = B[i + lda * j];
            }
        }
    }

    // For each block-row of A
    for (int j = 0; j < n; j += BLOCK_SIZE) {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            for (int i = 0; i < n; i += BLOCK_SIZE) {
                // For each block-column of B
                // Accumulate block dgemms into block of C
                // Perform individual block dgemm
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, n - i);
                int N = min(BLOCK_SIZE, n - j);
                int K = min(BLOCK_SIZE, n - k);
                do_block(n, M, N, K, A_trans + i + k * n, B_cpy + k + j * n, C_cpy + i + j * n);
            }
        }
    }

    for (int j=0; j<lda; j++) {
        for (int i=0; i<lda; i++) {
            C[i + lda * j] = C[i + lda * j] + C_cpy[i + n * j];
        }
    }

    free(A_trans);
    free(B_cpy);
    free(C_cpy);
}