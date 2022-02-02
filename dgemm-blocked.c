#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE 128
#endif
#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 512
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static inline __attribute__((optimize("unroll-loops"))) void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    __m512d cij, cij1, cij2, cij3, cij4, cij5, cij6, cij7;
    __m512d ci8j, ci8j1, ci8j2, ci8j3, ci8j4, ci8j5, ci8j6, ci8j7;
    __m512d aik, ai8k, bkj;
    for (int i=0; i<M; i+=16) {
        for (int j=0; j<N; j+=8) {
            double* Cij = C + (i + j * lda);
            double* Cij1 = C + (i + (j+1) * lda);
            double* Cij2 = C + (i + (j+2) * lda);
            double* Cij3 = C + (i + (j+3) * lda);
            double* Cij4 = C + (i + (j+4) * lda);
            double* Cij5 = C + (i + (j+5) * lda);
            double* Cij6 = C + (i + (j+6) * lda);
            double* Cij7 = C + (i + (j+7) * lda);
            double* Ci8j = C + (i + 8 + j * lda);
            double* Ci8j1 = C + (i + 8 + (j+1) * lda);
            double* Ci8j2 = C + (i + 8 + (j+2) * lda);
            double* Ci8j3 = C + (i + 8 + (j+3) * lda);
            double* Ci8j4 = C + (i + 8 + (j+4) * lda);
            double* Ci8j5 = C + (i + 8 + (j+5) * lda);
            double* Ci8j6 = C + (i + 8 + (j+6) * lda);
            double* Ci8j7 = C + (i + 8 + (j+7) * lda);
            asm volatile (
                "vmovapd (%[Cij]), %[cij]\n\t"
                "vmovapd (%[Cij1]), %[cij1]\n\t"
                "vmovapd (%[Cij2]), %[cij2]\n\t"
                "vmovapd (%[Cij3]), %[cij3]\n\t"
                "vmovapd (%[Cij4]), %[cij4]\n\t"
                "vmovapd (%[Cij5]), %[cij5]\n\t"
                "vmovapd (%[Cij6]), %[cij6]\n\t"
                "vmovapd (%[Cij7]), %[cij7]\n\t"
                : [cij] "+v" (cij),
                [cij1] "+v" (cij1),
                [cij2] "+v" (cij2),
                [cij3] "+v" (cij3),
                [cij4] "+v" (cij4),
                [cij5] "+v" (cij5),
                [cij6] "+v" (cij6),
                [cij7] "+v" (cij7)
                : [Cij] "r" (Cij),
                [Cij1] "r" (Cij1),
                [Cij2] "r" (Cij2),
                [Cij3] "r" (Cij3),
                [Cij4] "r" (Cij4),
                [Cij5] "r" (Cij5),
                [Cij6] "r" (Cij6),
                [Cij7] "r" (Cij7)
                : "memory"
            );
            asm volatile (
                "vmovapd (%[Ci8j]), %[ci8j]\n\t"
                "vmovapd (%[Ci8j1]), %[ci8j1]\n\t"
                "vmovapd (%[Ci8j2]), %[ci8j2]\n\t"
                "vmovapd (%[Ci8j3]), %[ci8j3]\n\t"
                "vmovapd (%[Ci8j4]), %[ci8j4]\n\t"
                "vmovapd (%[Ci8j5]), %[ci8j5]\n\t"
                "vmovapd (%[Ci8j6]), %[ci8j6]\n\t"
                "vmovapd (%[Ci8j7]), %[ci8j7]\n\t"
                : [ci8j] "+v" (ci8j),
                [ci8j1] "+v" (ci8j1),
                [ci8j2] "+v" (ci8j2),
                [ci8j3] "+v" (ci8j3),
                [ci8j4] "+v" (ci8j4),
                [ci8j5] "+v" (ci8j5),
                [ci8j6] "+v" (ci8j6),
                [ci8j7] "+v" (ci8j7)
                : [Ci8j] "r" (Ci8j),
                [Ci8j1] "r" (Ci8j1),
                [Ci8j2] "r" (Ci8j2),
                [Ci8j3] "r" (Ci8j3),
                [Ci8j4] "r" (Ci8j4),
                [Ci8j5] "r" (Ci8j5),
                [Ci8j6] "r" (Ci8j6),
                [Ci8j7] "r" (Ci8j7)
                : "memory"
            );
            for (int k=0; k<K; k+=1) {
                double* Aik = A + (i + k * lda);
                double* Ai8k = A + (i + 8 + k * lda);
                asm volatile (
                    "vmovapd (%[Aik]), %[aik]\n\t"
                    "vmovapd (%[Ai8k]), %[ai8k]\n\t"
                    "vfmadd231pd %[bkj], %[aik], %([cij])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j])"
                    "vfmadd231pd %[bkj], %[aik], %([cij1])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j1])"
                    "vfmadd231pd %[bkj], %[aik], %([cij2])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j2])"
                    "vfmadd231pd %[bkj], %[aik], %([cij3])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j3])"
                    "vfmadd231pd %[bkj], %[aik], %([cij4])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j4])"
                    "vfmadd231pd %[bkj], %[aik], %([cij5])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j5])"
                    "vfmadd231pd %[bkj], %[aik], %([cij6])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j6])"
                    "vfmadd231pd %[bkj], %[aik], %([cij7])"
                    "vfmadd231pd %[bkj], %[ai8k], %([ci8j7])"
                    : [aik] "+v" (aik),
                    [ai8k] "+v" (ai8k)
                    : [Aik] "r" (Aik),
                    [Ai8k] "r" (Ai8k),
                    [ai8k] "r" (ai8k),
                    [bkj] "r" (bkj),
                    [ci8j] "r" (ci8j),
                    [ci8j1] "r" (ci8j1),
                    [ci8j2] "r" (ci8j2),
                    [ci8j3] "r" (ci8j3),
                    [ci8j4] "r" (ci8j4),
                    [ci8j5] "r" (ci8j5),
                    [ci8j6] "r" (ci8j6),
                    [ci8j7] "r" (ci8j7)
                    : "memory"
                );
            }
            asm volatile (
                "vmovapd (%[cij]), %[Cij]"
                "vmovapd (%[cij1]), %[Cij1]"
                "vmovapd (%[cij2]), %[Cij2]"
                "vmovapd (%[cij3]), %[Cij3]"
                "vmovapd (%[cij4]), %[Cij4]"
                "vmovapd (%[cij5]), %[Cij5]"
                "vmovapd (%[cij6]), %[Cij6]"
                "vmovapd (%[cij7]), %[Cij7]"
                : [cij] "+v" (cij),
                [cij1] "+v" (cij1),
                [cij2] "+v" (cij2),
                [cij3] "+v" (cij3),
                [cij4] "+v" (cij4),
                [cij5] "+v" (cij5),
                [cij6] "+v" (cij6),
                [cij7] "+v" (cij7)
                : [Cij] "r" (Cij),
                [Cij1] "r" (Cij1),
                [Cij2] "r" (Cij2),
                [Cij3] "r" (Cij3),
                [Cij4] "r" (Cij4),
                [Cij5] "r" (Cij5),
                [Cij6] "r" (Cij6),
                [Cij7] "r" (Cij7)
                : "memory"
            );
            asm volatile (
                "vmovapd (%[ci8j]), %[Ci8j]"
                "vmovapd (%[ci8j1]), %[Ci8j1]"
                "vmovapd (%[ci8j2]), %[Ci8j2]"
                "vmovapd (%[ci8j3]), %[Ci8j3]"
                "vmovapd (%[ci8j4]), %[Ci8j4]"
                "vmovapd (%[ci8j5]), %[Ci8j5]"
                "vmovapd (%[ci8j6]), %[Ci8j6]"
                "vmovapd (%[ci8j7]), %[Ci8j7]"
                : [ci8j] "+v" (ci8j),
                [ci8j1] "+v" (ci8j1),
                [ci8j2] "+v" (ci8j2),
                [ci8j3] "+v" (ci8j3),
                [ci8j4] "+v" (ci8j4),
                [ci8j5] "+v" (ci8j5),
                [ci8j6] "+v" (ci8j6),
                [ci8j7] "+v" (ci8j7)
                : [Ci8j] "r" (Ci8j),
                [Ci8j1] "r" (Ci8j1),
                [Ci8j2] "r" (Ci8j2),
                [Ci8j3] "r" (Ci8j3),
                [Ci8j4] "r" (Ci8j4),
                [Ci8j5] "r" (Ci8j5),
                [Ci8j6] "r" (Ci8j6),
                [Ci8j7] "r" (Ci8j7)
                : "memory"
            );
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void l2_dgemm(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each block-row of A
    for (int j = 0; j < N; j += L1_BLOCK_SIZE) {
        for (int k = 0; k < K; k += L1_BLOCK_SIZE) {
            for (int i = 0; i < M; i += L1_BLOCK_SIZE) {
            // For each block-column of B
                // Accumulate block dgemms into block of C
                // Correct block dimensions if block "goes off edge of" the matrix
                int L1_M = min(L1_BLOCK_SIZE, M - i);
                int L1_N = min(L1_BLOCK_SIZE, N - j);
                int L1_K = min(L1_BLOCK_SIZE, K - k);
                // Perform individual block dgemm
                do_block(lda, L1_M, L1_N, L1_K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    int n = lda + (16 - (lda%16));

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
    for (int j = 0; j < n; j += L2_BLOCK_SIZE) {
        for (int k = 0; k < n; k += L2_BLOCK_SIZE) {
            for (int i = 0; i < n; i += L2_BLOCK_SIZE) {
            // For each block-column of B
                // Accumulate block dgemms into block of C
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(L2_BLOCK_SIZE, n - i);
                int N = min(L2_BLOCK_SIZE, n - j);
                int K = min(L2_BLOCK_SIZE, n - k);
                // Perform individual block dgemm
                l2_dgemm(n, M, N, K, A_trans + i + k * n, B_cpy + k + j * n, C_cpy + i + j * n);
            }
        }
    }

    for (int j=0; j<lda; j++) {
        for (int i=0; i<lda; i++) {
            C[i + lda * j] = C_cpy[i + n * j];
        }
    }

    free(A_trans);
    free(B_cpy);
    free(C_cpy);
}
