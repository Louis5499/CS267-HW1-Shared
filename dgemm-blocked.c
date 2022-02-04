#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif
#define min(a, b) (((a) < (b)) ? (a) : (b))


/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static inline __attribute__((optimize("unroll-loops"))) void asm_do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    __m512d cij, cij1, cij2, cij3, cij4, cij5, cij6, cij7;
    __m512d ci8j, ci8j1, ci8j2, ci8j3, ci8j4, ci8j5, ci8j6, ci8j7;
    __m512d aik, ai8k, bkj;
    for (int i=0; i<M; i+=16) {
        for (int j=0; j<N; j+=8) {
            double* Cij = C + (i + j * lda);
            asm volatile (
                "vmovapd 0(%[Cij]), %[cij]\n\t"
                "vmovapd 128(%[Cij]), %[cij1]\n\t"
                "vmovapd 256(%[Cij]), %[cij2]\n\t"
                "vmovapd 384(%[Cij]), %[cij3]\n\t"
                "vmovapd 512(%[Cij]), %[cij4]\n\t"
                "vmovapd 640(%[Cij]), %[cij5]\n\t"
                "vmovapd 768(%[Cij]), %[cij6]\n\t"
                "vmovapd 896(%[Cij]), %[cij7]\n\t"
                : [cij] "+v" (cij),
                [cij1] "+v" (cij1),
                [cij2] "+v" (cij2),
                [cij3] "+v" (cij3),
                [cij4] "+v" (cij4),
                [cij5] "+v" (cij5),
                [cij6] "+v" (cij6),
                [cij7] "+v" (cij7)
                : [Cij] "r" (Cij)
                : "memory"
            );
            asm volatile (
                "vmovapd 64(%[Cij]), %[ci8j]\n\t"
                "vmovapd 192(%[Cij]), %[ci8j1]\n\t"
                "vmovapd 320(%[Cij]), %[ci8j2]\n\t"
                "vmovapd 448(%[Cij]), %[ci8j3]\n\t"
                "vmovapd 576(%[Cij]), %[ci8j4]\n\t"
                "vmovapd 704(%[Cij]), %[ci8j5]\n\t"
                "vmovapd 832(%[Cij]), %[ci8j6]\n\t"
                "vmovapd 960(%[Cij]), %[ci8j7]\n\t"
                : [ci8j] "+v" (ci8j),
                [ci8j1] "+v" (ci8j1),
                [ci8j2] "+v" (ci8j2),
                [ci8j3] "+v" (ci8j3),
                [ci8j4] "+v" (ci8j4),
                [ci8j5] "+v" (ci8j5),
                [ci8j6] "+v" (ci8j6),
                [ci8j7] "+v" (ci8j7)
                : [Cij] "r" (Cij)
            );
            for (int k=0; k<K; k+=1) {
                double* Aik = A + (i + k * lda);
                double* Ai8k = A + (i + 8 + k * lda);
                double* Bkj = B + k + j * lda;
                asm volatile (
                    "vmovapd (%[Aik]), %[aik]\n\t"
                    "vfmadd231pd 0(%[Bkj])%{1to8%}, %[aik], %[cij]\n\t"
                    "vfmadd231pd 128(%[Bkj])%{1to8%}, %[aik], %[cij1]\n\t"
                    "vfmadd231pd 256(%[Bkj])%{1to8%}, %[aik], %[cij2]\n\t"
                    "vfmadd231pd 384(%[Bkj])%{1to8%}, %[aik], %[cij3]\n\t"
                    "vfmadd231pd 512(%[Bkj])%{1to8%}, %[aik], %[cij4]\n\t"
                    "vfmadd231pd 640(%[Bkj])%{1to8%}, %[aik], %[cij5]\n\t"
                    "vfmadd231pd 768(%[Bkj])%{1to8%}, %[aik], %[cij6]\n\t"
                    "vfmadd231pd 896(%[Bkj])%{1to8%}, %[aik], %[cij7]\n\t"
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
                asm volatile (
                    "vmovapd (%[Ai8k]), %[ai8k]\n\t"
                    "vfmadd231pd 0(%[Bkj])%{1to8%}, %[ai8k], %[ci8j]\n\t"
                    "vfmadd231pd 128(%[Bkj])%{1to8%}, %[ai8k], %[ci8j1]\n\t"
                    "vfmadd231pd 256(%[Bkj])%{1to8%}, %[ai8k], %[ci8j2]\n\t"
                    "vfmadd231pd 384(%[Bkj])%{1to8%}, %[ai8k], %[ci8j3]\n\t"
                    "vfmadd231pd 512(%[Bkj])%{1to8%}, %[ai8k], %[ci8j4]\n\t"
                    "vfmadd231pd 640(%[Bkj])%{1to8%}, %[ai8k], %[ci8j5]\n\t"
                    "vfmadd231pd 768(%[Bkj])%{1to8%}, %[ai8k], %[ci8j6]\n\t"
                    "vfmadd231pd 896(%[Bkj])%{1to8%}, %[ai8k], %[ci8j7]\n\t"
                    : [ci8j6] "+v" (ci8j6),
                    [ci8j4] "+v" (ci8j4),
                    [ci8j2] "+v" (ci8j2),
                    [ci8j5] "+v" (ci8j5),
                    [ci8j3] "+v" (ci8j3),
                    [ci8j] "+v" (ci8j),
                    [ci8j1] "+v" (ci8j1),
                    [ci8j7] "+v" (ci8j7),
                    [ai8k] "+v" (ai8k)
                    : [Bkj] "r" (Bkj),
                    [Ai8k] "r" (Ai8k)
                );
            }
            asm volatile (
                "vmovapd %[cij], 0(%[Cij])\n\t"
                "vmovapd %[cij1], 128(%[Cij])\n\t"
                "vmovapd %[cij2], 256(%[Cij])\n\t"
                "vmovapd %[cij3], 384(%[Cij])\n\t"
                "vmovapd %[cij4], 512(%[Cij])\n\t"
                "vmovapd %[cij5], 640(%[Cij])\n\t"
                "vmovapd %[cij6], 768(%[Cij])\n\t"
                "vmovapd %[cij7], 896(%[Cij])\n\t"
                : [cij3] "+v" (cij3),
                [cij4] "+v" (cij4),
                [cij6] "+v" (cij6),
                [cij2] "+v" (cij2),
                [cij5] "+v" (cij5),
                [cij1] "+v" (cij1),
                [cij] "+v" (cij),
                [cij7] "+v" (cij7)
                : [Cij] "r" (Cij)
                : "memory"
            );
            asm volatile (
                "vmovapd %[ci8j], 64(%[Cij])\n\t"
                "vmovapd %[ci8j1], 192(%[Cij])\n\t"
                "vmovapd %[ci8j2], 320(%[Cij])\n\t"
                "vmovapd %[ci8j3], 448(%[Cij])\n\t"
                "vmovapd %[ci8j4], 576(%[Cij])\n\t"
                "vmovapd %[ci8j5], 704(%[Cij])\n\t"
                "vmovapd %[ci8j6], 832(%[Cij])\n\t"
                "vmovapd %[ci8j7], 960(%[Cij])\n\t"
                : [ci8j3] "+v" (ci8j3),
                [ci8j4] "+v" (ci8j4),
                [ci8j6] "+v" (ci8j6),
                [ci8j2] "+v" (ci8j2),
                [ci8j5] "+v" (ci8j5),
                [ci8j1] "+v" (ci8j1),
                [ci8j] "+v" (ci8j),
                [ci8j7] "+v" (ci8j7)
                : [Cij] "r" (Cij)
                : "memory"
            );
        }
    }
}

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
            cij = _mm512_loadu_pd(C + (i + j * lda));
            cij1 = _mm512_loadu_pd(C + (i + (j+1) * lda));
            cij2 = _mm512_loadu_pd(C + (i + (j+2) * lda));
            cij3 = _mm512_loadu_pd(C + (i + (j+3) * lda));
            cij4 = _mm512_loadu_pd(C + (i + (j+4) * lda));
            cij5 = _mm512_loadu_pd(C + (i + (j+5) * lda));
            cij6 = _mm512_loadu_pd(C + (i + (j+6) * lda));
            cij7 = _mm512_loadu_pd(C + (i + (j+7) * lda));

            ci8j = _mm512_loadu_pd(C + (i + 8 + j * lda));
            ci8j1 = _mm512_loadu_pd(C + (i + 8 + (j+1) * lda));
            ci8j2 = _mm512_loadu_pd(C + (i + 8 + (j+2) * lda));
            ci8j3 = _mm512_loadu_pd(C + (i + 8 + (j+3) * lda));
            ci8j4 = _mm512_loadu_pd(C + (i + 8 + (j+4) * lda));
            ci8j5 = _mm512_loadu_pd(C + (i + 8 + (j+5) * lda));
            ci8j6 = _mm512_loadu_pd(C + (i + 8 + (j+6) * lda));
            ci8j7 = _mm512_loadu_pd(C + (i + 8 + (j+7) * lda));

            for (int k=0; k<K; k+=1) {
                aik = _mm512_loadu_pd(A + (i + k * lda));
                ai8k = _mm512_loadu_pd(A + (i + 8 + k * lda));

                bkj = _mm512_set1_pd(B[k + j * lda]);
                cij = _mm512_fmadd_pd(aik, bkj, cij);
                ci8j = _mm512_fmadd_pd(ai8k, bkj, ci8j);

                bkj = _mm512_set1_pd(B[k + (j+1) * lda]);
                cij1 = _mm512_fmadd_pd(aik, bkj, cij1);
                ci8j1 = _mm512_fmadd_pd(ai8k, bkj, ci8j1);

                bkj = _mm512_set1_pd(B[k + (j+2) * lda]);
                cij2 = _mm512_fmadd_pd(aik, bkj, cij2);
                ci8j2 = _mm512_fmadd_pd(ai8k, bkj, ci8j2);

                bkj = _mm512_set1_pd(B[k + (j+3) * lda]);
                cij3 = _mm512_fmadd_pd(aik, bkj, cij3);
                ci8j3 = _mm512_fmadd_pd(ai8k, bkj, ci8j3);

                bkj = _mm512_set1_pd(B[k + (j+4) * lda]);
                cij4 = _mm512_fmadd_pd(aik, bkj, cij4);
                ci8j4 = _mm512_fmadd_pd(ai8k, bkj, ci8j4);

                bkj = _mm512_set1_pd(B[k + (j+5) * lda]);
                cij5 = _mm512_fmadd_pd(aik, bkj, cij5);
                ci8j5 = _mm512_fmadd_pd(ai8k, bkj, ci8j5);

                bkj = _mm512_set1_pd(B[k + (j+6) * lda]);
                cij6 = _mm512_fmadd_pd(aik, bkj, cij6);
                ci8j6 = _mm512_fmadd_pd(ai8k, bkj, ci8j6);

                bkj = _mm512_set1_pd(B[k + (j+7) * lda]);
                cij7 = _mm512_fmadd_pd(aik, bkj, cij7);
                ci8j7 = _mm512_fmadd_pd(ai8k, bkj, ci8j7);

            }
            _mm512_store_pd(C + i + j * lda, cij);
            _mm512_store_pd(C + i + (j+1) * lda, cij1);
            _mm512_store_pd(C + i + (j+2) * lda, cij2);
            _mm512_store_pd(C + i + (j+3) * lda, cij3);
            _mm512_store_pd(C + i + (j+4) * lda, cij4);
            _mm512_store_pd(C + i + (j+5) * lda, cij5);
            _mm512_store_pd(C + i + (j+6) * lda, cij6);
            _mm512_store_pd(C + i + (j+7) * lda, cij7);


            _mm512_store_pd(C + i + 8 + j * lda, ci8j);
            _mm512_store_pd(C + i + 8 + (j+1) * lda, ci8j1);
            _mm512_store_pd(C + i + 8 + (j+2) * lda, ci8j2);
            _mm512_store_pd(C + i + 8 + (j+3) * lda, ci8j3);
            _mm512_store_pd(C + i + 8 + (j+4) * lda, ci8j4);
            _mm512_store_pd(C + i + 8 + (j+5) * lda, ci8j5);
            _mm512_store_pd(C + i + 8 + (j+6) * lda, ci8j6);
            _mm512_store_pd(C + i + 8 + (j+7) * lda, ci8j7);
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
    for (int j = 0; j < n; j += BLOCK_SIZE) {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            for (int i = 0; i < n; i += BLOCK_SIZE) {
            // For each block-column of B
                // Accumulate block dgemms into block of C
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, n - i);
                int N = min(BLOCK_SIZE, n - j);
                int K = min(BLOCK_SIZE, n - k);
                // Perform individual block dgemm
                asm_do_block(n, M, N, K, A_trans + i + k * n, B_cpy + k + j * n, C_cpy + i + j * n);
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