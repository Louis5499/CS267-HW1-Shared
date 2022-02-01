#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
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
                do_block(n, M, N, K, A_trans + i + k * n, B_cpy + k + j * n, C_cpy + i + j * n);
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
