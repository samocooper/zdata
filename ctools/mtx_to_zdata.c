#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <float.h>
#include <math.h>

#include "zstd.h"
#include "zstd_seekable.h"   /* from contrib/seekable_format */

/* Default values (can be overridden via command line) */
#define DEFAULT_BLOCK_ROWS 16
#define DEFAULT_MAX_ROWS   8192

/* -----------------------------------------------------------------------
   Abstract dtype system: version ↔ element size ↔ name
   ----------------------------------------------------------------------- */
/* DTypeInfo and DTYPE_TABLE are generated from zdata/dtypes.py -- the single
   source of truth shared with the Python layer. Regenerate with:
       python -m zdata.dtypes --write-header                                  */
#include "dtype_table.h"

#define NUM_DTYPES ZDATA_NUM_DTYPES

/* Currently selected dtype (default: uint16, version 2) */
static const DTypeInfo *g_dtype = &DTYPE_TABLE[0];

/* When set, the COO payload after the size line is raw binary (struct-of-
   arrays): nnz int32 row indices, nnz int32 col indices, nnz values of the
   selected dtype – all 0-based, little-endian.  Used by the Python streaming
   pipeline to avoid materialising text MTX files on disk. */
static int g_binary_input = 0;

/* zstd compression level for the seekable archives (1 = fastest/largest).
   Overridable with --level; the format is seekable at any level. */
static int g_zstd_level = 1;

static const DTypeInfo *dtype_by_name(const char *name) {
    for (size_t i = 0; i < NUM_DTYPES; i++)
        if (strcmp(DTYPE_TABLE[i].name, name) == 0) return &DTYPE_TABLE[i];
    return NULL;
}

static const DTypeInfo *dtype_by_version(uint32_t ver) {
    for (size_t i = 0; i < NUM_DTYPES; i++)
        if (DTYPE_TABLE[i].version == ver) return &DTYPE_TABLE[i];
    return NULL;
}

/* ---------------------------------------------------------------------------
   IEEE-754 binary16 ("half") encoding.

   Uses the compiler's native _Float16 where available (GCC 12+, Clang 15+ on
   common targets); otherwise falls back to an explicit bit-level conversion so
   the on-disk bytes are identical either way. Both paths round-to-nearest and
   saturate to +/-Inf on overflow, matching numpy's float16 cast.
   --------------------------------------------------------------------------- */
#if defined(__FLT16_MAX__) && !defined(ZDATA_NO_NATIVE_FLOAT16)
#  define ZDATA_HAVE_NATIVE_FLOAT16 1
#endif

static uint16_t float_to_half_bits(double value) {
#ifdef ZDATA_HAVE_NATIVE_FLOAT16
    _Float16 h = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &h, sizeof bits);
    return bits;
#else
    float f = (float)value;
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t expo = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (expo == 0xFFu)                       /* Inf / NaN */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));

    int32_t e = (int32_t)expo - 127 + 15;    /* rebias 127 -> 15 */
    if (e >= 0x1F)                           /* overflow -> Inf */
        return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {                            /* subnormal or zero */
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;                   /* restore implicit leading 1 */
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) h += 1u;   /* round half up */
        return (uint16_t)(sign | h);
    }
    {
        uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
        if (mant & 0x1000u) h += 1u;         /* round half up (may carry) */
        return (uint16_t)(sign | h);
    }
#endif
}

/* Convert a double value (read from MTX text) to the target dtype and
   store it at *dst.  dst must point to val_size bytes of writable memory. */
static void store_value(void *dst, double value, const DTypeInfo *dt) {
    switch (dt->version) {
    case 2: { /* uint16 */
        double c = value;
        if (c < 0.0) c = 0.0;
        if (c > (double)UINT16_MAX) c = (double)UINT16_MAX;
        *(uint16_t *)dst = (uint16_t)(c + 0.5);
        break;
    }
    case 3: /* float32 */
        *(float *)dst = (float)value;
        break;
    case 4: { /* uint8 */
        double c = value;
        if (c < 0.0) c = 0.0;
        if (c > 255.0) c = 255.0;
        *(uint8_t *)dst = (uint8_t)(c + 0.5);
        break;
    }
    case 5: { /* uint32 */
        double c = value;
        if (c < 0.0) c = 0.0;
        if (c > (double)UINT32_MAX) c = (double)UINT32_MAX;
        *(uint32_t *)dst = (uint32_t)(c + 0.5);
        break;
    }
    case 6: { /* uint64 */
        double c = value;
        if (c < 0.0) c = 0.0;
        if (c > (double)UINT64_MAX) c = (double)UINT64_MAX;
        *(uint64_t *)dst = (uint64_t)(c + 0.5);
        break;
    }
    case 7: { /* int8 */
        double c = value;
        if (c < -128.0) c = -128.0;
        if (c > 127.0) c = 127.0;
        *(int8_t *)dst = (int8_t)(c < 0 ? c - 0.5 : c + 0.5);
        break;
    }
    case 8: { /* int16 */
        double c = value;
        if (c < (double)INT16_MIN) c = (double)INT16_MIN;
        if (c > (double)INT16_MAX) c = (double)INT16_MAX;
        *(int16_t *)dst = (int16_t)(c < 0 ? c - 0.5 : c + 0.5);
        break;
    }
    case 9: { /* int32 */
        double c = value;
        if (c < (double)INT32_MIN) c = (double)INT32_MIN;
        if (c > (double)INT32_MAX) c = (double)INT32_MAX;
        *(int32_t *)dst = (int32_t)(c < 0 ? c - 0.5 : c + 0.5);
        break;
    }
    case 10: { /* int64 */
        double c = value;
        if (c < (double)INT64_MIN) c = (double)INT64_MIN;
        if (c > (double)INT64_MAX) c = (double)INT64_MAX;
        *(int64_t *)dst = (int64_t)c;
        break;
    }
    case 11: /* float64 */
        *(double *)dst = value;
        break;
    case 12: { /* float16 (IEEE-754 binary16) */
        uint16_t bits = float_to_half_bits(value);
        memcpy(dst, &bits, sizeof bits);
        break;
    }
    default:
        /* Fallback: treat as uint16 */
        *(uint16_t *)dst = (uint16_t)(value + 0.5);
        break;
    }
}

/* Store the default value (1) for entries with no explicit value in MTX. */
static void store_default_value(void *dst, const DTypeInfo *dt) {
    store_value(dst, 1.0, dt);
}

/* -----------------------------------------------------------------------
   Block-CSR structures – type-agnostic via void* + val_size
   ----------------------------------------------------------------------- */
typedef struct {
    uint32_t  nnz;
    uint32_t *indptr;     /* length block_rows+1 */
    uint32_t *indices;    /* length nnz */
    void     *data;       /* length nnz * val_size */
    uint32_t *write_pos;
} BlockCSR;

typedef struct {
    /* indptr is uint64 because the *total* nnz of a full matrix passed to the
       compressor (a streamed X_CM gene slab can hold billions) routinely
       exceeds 2^32. Per-block indptrs in the on-disk format stay uint32 –
       blocks are bounded to (block_rows * ncols) entries. */
    uint64_t  *indptr;
    uint32_t  *indices;
    void      *data;      /* length nnz_total * val_size */
    long long  nnz_total;
} FullCSR;

/* -----------------------------------------------------------------------
   Helpers
   ----------------------------------------------------------------------- */
static int skip_to_size_line(FILE *f, char *line, size_t line_sz) {
    do {
        if (!fgets(line, (int)line_sz, f)) return 0;
    } while (line[0] == '%');
    return 1;
}

static void write_le32(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xFF);
    dst[1] = (uint8_t)((v >> 8) & 0xFF);
    dst[2] = (uint8_t)((v >> 16) & 0xFF);
    dst[3] = (uint8_t)((v >> 24) & 0xFF);
}

/* -----------------------------------------------------------------------
   Serialize a CSR block – fully type-agnostic via val_size
   ----------------------------------------------------------------------- */
static uint8_t* serialize_block(
    const BlockCSR *b,
    uint32_t start_row,
    uint32_t nrows_in_block,
    uint32_t ncols,
    uint32_t block_rows,
    size_t *out_size
) {
    size_t val_size = g_dtype->val_size;

    size_t header_bytes  = 6 * 4;
    size_t indptr_bytes  = (block_rows + 1) * sizeof(uint32_t);
    size_t indices_bytes = (size_t)b->nnz * sizeof(uint32_t);
    size_t data_bytes    = (size_t)b->nnz * val_size;

    size_t total = header_bytes + indptr_bytes + indices_bytes + data_bytes;
    uint8_t *buf = (uint8_t*)malloc(total);
    if (!buf) return NULL;

    size_t off = 0;
    write_le32(buf + off, 0x5253435A); off += 4;          /* magic 'ZCSR' */
    write_le32(buf + off, g_dtype->version); off += 4;     /* version      */
    write_le32(buf + off, start_row);        off += 4;
    write_le32(buf + off, nrows_in_block);   off += 4;
    write_le32(buf + off, ncols);            off += 4;
    write_le32(buf + off, (uint32_t)b->nnz); off += 4;

    memcpy(buf + off, b->indptr, indptr_bytes);  off += indptr_bytes;
    memcpy(buf + off, b->indices, indices_bytes); off += indices_bytes;
    memcpy(buf + off, b->data, data_bytes);       off += data_bytes;

    *out_size = total;
    return buf;
}

/* -----------------------------------------------------------------------
   Seekable-ZSTD helpers
   ----------------------------------------------------------------------- */
static int flush_seekable_to_file(ZSTD_seekable_CStream *zcs, FILE *out,
                                   const void *src, size_t srcSize) {
    ZSTD_inBuffer inb = { src, srcSize, 0 };
    uint8_t outbuf[1 << 20];
    while (inb.pos < inb.size) {
        ZSTD_outBuffer outb = { outbuf, sizeof(outbuf), 0 };
        size_t r = ZSTD_seekable_compressStream(zcs, &outb, &inb);
        if (ZSTD_isError(r)) {
            fprintf(stderr, "ZSTD compress error: %s\n", ZSTD_getErrorName(r));
            return 0;
        }
        if (outb.pos && fwrite(outbuf, 1, outb.pos, out) != outb.pos) {
            perror("fwrite"); return 0;
        }
    }
    return 1;
}

static int end_frame_to_file(ZSTD_seekable_CStream *zcs, FILE *out) {
    uint8_t outbuf[1 << 20];
    for (;;) {
        ZSTD_outBuffer outb = { outbuf, sizeof(outbuf), 0 };
        size_t r = ZSTD_seekable_endFrame(zcs, &outb);
        if (ZSTD_isError(r)) {
            fprintf(stderr, "endFrame error: %s\n", ZSTD_getErrorName(r));
            return 0;
        }
        if (outb.pos && fwrite(outbuf, 1, outb.pos, out) != outb.pos) {
            perror("fwrite"); return 0;
        }
        if (r == 0) break;
    }
    return 1;
}

static int end_stream_to_file(ZSTD_seekable_CStream *zcs, FILE *out) {
    uint8_t outbuf[1 << 20];
    for (;;) {
        ZSTD_outBuffer outb = { outbuf, sizeof(outbuf), 0 };
        size_t r = ZSTD_seekable_endStream(zcs, &outb);
        if (ZSTD_isError(r)) {
            fprintf(stderr, "endStream error: %s\n", ZSTD_getErrorName(r));
            return 0;
        }
        if (outb.pos && fwrite(outbuf, 1, outb.pos, out) != outb.pos) {
            perror("fwrite"); return 0;
        }
        if (r == 0) break;
    }
    return 1;
}

/* -----------------------------------------------------------------------
   Row accumulator (used during MTX parsing)
   ----------------------------------------------------------------------- */
typedef struct {
    uint32_t *indices;
    void     *data;      /* val_size bytes per element */
    uint32_t  size;
    uint32_t  capacity;
} RowEntries;

/* Fast manual MTX line parser */
static int parse_line_fast(const char *line, long long *row, long long *col, double *value) {
    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n') return 0;

    *row = 0;
    while (*p >= '0' && *p <= '9') { *row = *row * 10 + (*p - '0'); p++; }
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n') return 1;

    *col = 0;
    while (*p >= '0' && *p <= '9') { *col = *col * 10 + (*p - '0'); p++; }
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n') return 2;

    /* Parse value */
    *value = 0.0;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') { p++; }

    while (*p >= '0' && *p <= '9') { *value = *value * 10.0 + (*p - '0'); p++; }
    if (*p == '.') {
        p++;
        double frac = 0.1;
        while (*p >= '0' && *p <= '9') { *value += (*p - '0') * frac; frac *= 0.1; p++; }
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        int esign = 1;
        if (*p == '-') { esign = -1; p++; } else if (*p == '+') { p++; }
        int exp = 0;
        while (*p >= '0' && *p <= '9') { exp = exp * 10 + (*p - '0'); p++; }
        for (int i = 0; i < exp; i++) { if (esign > 0) *value *= 10.0; else *value *= 0.1; }
    }
    *value *= sign;
    return 3;
}

/* -----------------------------------------------------------------------
   Build FullCSR from MTX file – type-agnostic
   ----------------------------------------------------------------------- */
static FullCSR* build_full_csr(FILE *f, long long nrows, long long ncols) {
    /* Reads sequentially from the current file position (just past the size
       line).  Works on regular files and on non-seekable streams (stdin/FIFO)
       alike – no fseek, so COO triplets can be piped straight in. */
    size_t vs = g_dtype->val_size;

    RowEntries *rows = (RowEntries*)calloc((size_t)nrows, sizeof(RowEntries));
    if (!rows) { fprintf(stderr, "OOM rows\n"); return NULL; }

    const uint32_t INIT_CAP = 64;
    for (long long i = 0; i < nrows; i++) {
        rows[i].capacity = INIT_CAP;
        rows[i].indices = (uint32_t*)malloc(INIT_CAP * sizeof(uint32_t));
        rows[i].data    = malloc(INIT_CAP * vs);
        if (!rows[i].indices || !rows[i].data) {
            fprintf(stderr, "OOM initial row arrays\n");
            for (long long j = 0; j <= i; j++) { free(rows[j].indices); free(rows[j].data); }
            free(rows);
            return NULL;
        }
        rows[i].size = 0;
    }

    char line[8192];
    long long row, col;
    double value = 0.0;
    long long lines_processed = 0, last_progress = 0;

    printf("  Building CSR structure (single pass, optimized)...\n");
    fflush(stdout);

    while (fgets(line, sizeof(line), f)) {
        int n = parse_line_fast(line, &row, &col, &value);
        if (n < 2) continue;

        lines_processed++;
        if (lines_processed - last_progress >= 50000000) {
            printf("    Processed %lld lines...\n", lines_processed);
            fflush(stdout);
            last_progress = lines_processed;
        }

        long long local_row = row - 1;
        col -= 1;
        if (local_row < 0 || local_row >= nrows || col < 0 || col >= ncols) continue;

        RowEntries *r = &rows[local_row];

        /* Grow if needed */
        if (r->size >= r->capacity) {
            uint32_t nc = r->capacity * 2;
            uint32_t *ni = (uint32_t*)realloc(r->indices, nc * sizeof(uint32_t));
            void     *nd = realloc(r->data, nc * vs);
            if (!ni || !nd) {
                fprintf(stderr, "OOM realloc row arrays\n");
                for (long long j = 0; j < nrows; j++) { free(rows[j].indices); free(rows[j].data); }
                free(rows);
                return NULL;
            }
            r->indices = ni;
            r->data    = nd;
            r->capacity = nc;
        }

        r->indices[r->size] = (uint32_t)col;
        if (n == 3)
            store_value((uint8_t*)r->data + r->size * vs, value, g_dtype);
        else
            store_default_value((uint8_t*)r->data + r->size * vs, g_dtype);
        r->size++;
    }

    if (lines_processed >= 50000000) {
        printf("    Completed reading: %lld total lines processed\n", lines_processed);
        fflush(stdout);
    }

    /* Build contiguous CSR */
    printf("  Converting to CSR format...\n");
    fflush(stdout);

    uint64_t *indptr = (uint64_t*)malloc((size_t)(nrows + 1) * sizeof(uint64_t));
    if (!indptr) { fprintf(stderr, "OOM indptr\n"); goto fail_rows; }

    indptr[0] = 0;
    long long nnz_total = 0;
    for (long long i = 0; i < nrows; i++) {
        nnz_total += rows[i].size;
        indptr[i + 1] = (uint64_t)nnz_total;
    }

    if (nnz_total == 0) {
        fprintf(stderr, "Error: matrix has no non-zero elements\n");
        free(indptr); goto fail_rows;
    }

    uint32_t *indices = (uint32_t*)malloc((size_t)nnz_total * sizeof(uint32_t));
    void     *data    = malloc((size_t)nnz_total * vs);
    if (!indices || !data) {
        fprintf(stderr, "OOM indices/data\n");
        free(indptr); if (indices) free(indices); if (data) free(data);
        goto fail_rows;
    }

    uint32_t pos = 0;
    for (long long i = 0; i < nrows; i++) {
        if (rows[i].size > 0) {
            memcpy(indices + pos, rows[i].indices, rows[i].size * sizeof(uint32_t));
            memcpy((uint8_t*)data + pos * vs, rows[i].data, rows[i].size * vs);
            pos += rows[i].size;
        }
        free(rows[i].indices);
        free(rows[i].data);
    }
    free(rows);

    FullCSR *csr = (FullCSR*)malloc(sizeof(FullCSR));
    if (!csr) { fprintf(stderr, "OOM FullCSR\n"); free(indptr); free(indices); free(data); return NULL; }
    csr->indptr = indptr;
    csr->indices = indices;
    csr->data = data;
    csr->nnz_total = nnz_total;

    printf("  CSR structure complete (%lld nnz)\n", nnz_total);
    fflush(stdout);
    return csr;

fail_rows:
    for (long long i = 0; i < nrows; i++) { free(rows[i].indices); free(rows[i].data); }
    free(rows);
    return NULL;
}

/* -----------------------------------------------------------------------
   Build FullCSR from a raw binary COO stream (struct-of-arrays):
     int32 row[nnz], int32 col[nnz], <dtype> val[nnz]   – all 0-based.
   Uses a counting sort straight into CSR (no per-row realloc growth).
   ----------------------------------------------------------------------- */
static FullCSR* build_full_csr_binary(FILE *f, long long nrows, long long ncols,
                                      long long nnz_in) {
    size_t vs = g_dtype->val_size;
    if (nnz_in <= 0) { fprintf(stderr, "Error: nnz must be > 0\n"); return NULL; }

    int32_t *all_r = (int32_t*)malloc((size_t)nnz_in * sizeof(int32_t));
    int32_t *all_c = (int32_t*)malloc((size_t)nnz_in * sizeof(int32_t));
    void    *all_v = malloc((size_t)nnz_in * vs);
    if (!all_r || !all_c || !all_v) {
        fprintf(stderr, "OOM reading binary COO stream\n");
        free(all_r); free(all_c); free(all_v); return NULL;
    }

    printf("  Reading binary COO stream (%lld triplets)...\n", nnz_in);
    fflush(stdout);
    if (fread(all_r, sizeof(int32_t), (size_t)nnz_in, f) != (size_t)nnz_in ||
        fread(all_c, sizeof(int32_t), (size_t)nnz_in, f) != (size_t)nnz_in ||
        fread(all_v, vs,              (size_t)nnz_in, f) != (size_t)nnz_in) {
        fprintf(stderr, "Short read on binary COO stream\n");
        free(all_r); free(all_c); free(all_v); return NULL;
    }

    /* Count nnz per row (skipping out-of-range entries).  64-bit indptr is
       required: a streamed CM gene-slab routinely holds >2^32 nnz. */
    uint64_t *indptr = (uint64_t*)calloc((size_t)(nrows + 1), sizeof(uint64_t));
    if (!indptr) { fprintf(stderr, "OOM indptr\n"); free(all_r); free(all_c); free(all_v); return NULL; }
    for (long long i = 0; i < nnz_in; i++) {
        int32_t r = all_r[i], c = all_c[i];
        if (r >= 0 && r < nrows && c >= 0 && (long long)c < ncols) indptr[r + 1]++;
    }
    for (long long i = 0; i < nrows; i++) indptr[i + 1] += indptr[i];
    long long nnz_total = (long long)indptr[nrows];
    if (nnz_total == 0) {
        fprintf(stderr, "Error: matrix has no in-range non-zero elements\n");
        free(indptr); free(all_r); free(all_c); free(all_v); return NULL;
    }

    uint32_t *indices = (uint32_t*)malloc((size_t)nnz_total * sizeof(uint32_t));
    void     *data    = malloc((size_t)nnz_total * vs);
    uint64_t *wpos    = (uint64_t*)malloc((size_t)nrows * sizeof(uint64_t));
    if (!indices || !data || !wpos) {
        fprintf(stderr, "OOM CSR arrays\n");
        free(indptr); free(indices); free(data); free(wpos);
        free(all_r); free(all_c); free(all_v); return NULL;
    }
    for (long long i = 0; i < nrows; i++) wpos[i] = indptr[i];

    /* Place entries (input order is row-major / col-sorted within row) */
    for (long long i = 0; i < nnz_in; i++) {
        int32_t r = all_r[i], c = all_c[i];
        if (r < 0 || r >= nrows || c < 0 || (long long)c >= ncols) continue;
        uint64_t p = wpos[r]++;
        indices[p] = (uint32_t)c;
        memcpy((uint8_t*)data + (size_t)p * vs, (uint8_t*)all_v + (size_t)i * vs, vs);
    }

    free(all_r); free(all_c); free(all_v); free(wpos);

    FullCSR *csr = (FullCSR*)malloc(sizeof(FullCSR));
    if (!csr) { fprintf(stderr, "OOM FullCSR\n"); free(indptr); free(indices); free(data); return NULL; }
    csr->indptr = indptr;
    csr->indices = indices;
    csr->data = data;
    csr->nnz_total = nnz_total;
    printf("  CSR structure complete (%lld nnz)\n", nnz_total);
    fflush(stdout);
    return csr;
}

static void free_full_csr(FullCSR *csr) {
    if (csr) {
        free(csr->indptr);
        free(csr->indices);
        free(csr->data);
        free(csr);
    }
}

/* -----------------------------------------------------------------------
   Process a chunk of rows from CSR – type-agnostic
   ----------------------------------------------------------------------- */
static int process_chunk_from_csr(
    const FullCSR *csr,
    long long start_row_global,
    long long end_row_global,
    const char *out_path,
    uint32_t block_rows,
    uint32_t max_rows,
    long long row_offset,
    uint32_t ncols
) {
    size_t vs = g_dtype->val_size;
    uint32_t chunk_size = (uint32_t)(end_row_global - start_row_global);
    if (chunk_size == 0) return 1;
    if (chunk_size > max_rows) chunk_size = max_rows;

    long long start_row_local = start_row_global - row_offset;

    uint32_t *row_counts = (uint32_t*)malloc(chunk_size * sizeof(uint32_t));
    if (!row_counts) { fprintf(stderr, "OOM row_counts\n"); return 0; }
    for (uint32_t i = 0; i < chunk_size; i++) {
        long long ri = start_row_local + i;
        row_counts[i] = (uint32_t)(csr->indptr[ri + 1] - csr->indptr[ri]);
    }

    uint32_t nblocks = (chunk_size + block_rows - 1) / block_rows;
    BlockCSR *blocks = (BlockCSR*)calloc(nblocks, sizeof(BlockCSR));
    if (!blocks) { fprintf(stderr, "OOM blocks\n"); free(row_counts); return 0; }

    for (uint32_t b = 0; b < nblocks; b++) {
        blocks[b].indptr = (uint32_t*)malloc((block_rows + 1) * sizeof(uint32_t));
        blocks[b].write_pos = (uint32_t*)malloc(block_rows * sizeof(uint32_t));
        if (!blocks[b].indptr || !blocks[b].write_pos) { fprintf(stderr, "OOM\n"); return 0; }

        blocks[b].indptr[0] = 0;
        for (uint32_t r = 0; r < block_rows; r++) {
            uint32_t lr = b * block_rows + r;
            blocks[b].indptr[r + 1] = blocks[b].indptr[r] +
                ((lr < chunk_size) ? row_counts[lr] : 0);
        }
        blocks[b].nnz = blocks[b].indptr[block_rows];
        if (blocks[b].nnz > 0) {
            blocks[b].indices = (uint32_t*)malloc((size_t)blocks[b].nnz * sizeof(uint32_t));
            blocks[b].data    = malloc((size_t)blocks[b].nnz * vs);
            if (!blocks[b].indices || !blocks[b].data) { fprintf(stderr, "OOM\n"); return 0; }
        }
        for (uint32_t r = 0; r < block_rows; r++)
            blocks[b].write_pos[r] = blocks[b].indptr[r];
    }

    /* Fill blocks (uint64 rs/re because the global FullCSR can hold >2^32 nnz;
       per-block write_pos stays uint32 since one block holds <= block_rows*ncols
       entries) */
    for (uint32_t i = 0; i < chunk_size; i++) {
        long long ri = start_row_local + i;
        uint64_t rs = csr->indptr[ri], re = csr->indptr[ri + 1];
        uint32_t b = i / block_rows, r = i % block_rows;
        uint32_t row_nnz = (uint32_t)(re - rs);
        for (uint32_t j = 0; j < row_nnz; j++) {
            uint32_t p = blocks[b].write_pos[r]++;
            blocks[b].indices[p] = csr->indices[rs + j];
            memcpy((uint8_t*)blocks[b].data + (size_t)p * vs,
                   (uint8_t*)csr->data + (rs + j) * vs, vs);
        }
    }

    /* Write seekable archive */
    FILE *out = fopen(out_path, "wb");
    if (!out) { perror("open out"); return 0; }
    setvbuf(out, NULL, _IOFBF, 256 * 1024);

    ZSTD_seekable_CStream *zcs = ZSTD_seekable_createCStream();
    if (!zcs) { fprintf(stderr, "createCStream failed\n"); fclose(out); return 0; }
    size_t initR = ZSTD_seekable_initCStream(zcs, g_zstd_level, 0, 0);
    if (ZSTD_isError(initR)) {
        fprintf(stderr, "initCStream error: %s\n", ZSTD_getErrorName(initR));
        ZSTD_seekable_freeCStream(zcs); fclose(out); return 0;
    }

    for (uint32_t b = 0; b < nblocks; b++) {
        uint32_t sr = (uint32_t)start_row_global + b * block_rows;
        uint32_t nr = (b == nblocks - 1) ? (chunk_size - b * block_rows) : block_rows;
        size_t blob_sz = 0;
        uint8_t *blob = serialize_block(&blocks[b], sr, nr, ncols, block_rows, &blob_sz);
        if (!blob) { fprintf(stderr, "OOM serialize\n"); ZSTD_seekable_freeCStream(zcs); fclose(out); return 0; }
        if (!flush_seekable_to_file(zcs, out, blob, blob_sz)) { free(blob); ZSTD_seekable_freeCStream(zcs); fclose(out); return 0; }
        free(blob);
        if (!end_frame_to_file(zcs, out)) { ZSTD_seekable_freeCStream(zcs); fclose(out); return 0; }
    }

    if (!end_stream_to_file(zcs, out)) { ZSTD_seekable_freeCStream(zcs); fclose(out); return 0; }
    ZSTD_seekable_freeCStream(zcs);
    fclose(out);

    for (uint32_t b = 0; b < nblocks; b++) {
        free(blocks[b].indptr);
        free(blocks[b].indices);
        free(blocks[b].data);
        free(blocks[b].write_pos);
    }
    free(blocks);
    free(row_counts);
    return 1;
}


int main(int argc, char *argv[]) {
    /* Parse optional flags before positional args */
    /* Supported: --float32  (backward compat, same as --dtype float32)
                  --dtype <name>  */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--float32") == 0) {
            g_dtype = dtype_by_name("float32");
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--; i--;
        } else if (strcmp(argv[i], "--binary") == 0) {
            g_binary_input = 1;
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--; i--;
        } else if (strcmp(argv[i], "--level") == 0 && i + 1 < argc) {
            g_zstd_level = atoi(argv[i + 1]);
            if (g_zstd_level < 1) g_zstd_level = 1;
            if (g_zstd_level > 22) g_zstd_level = 22;
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2; i--;
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            const DTypeInfo *dt = dtype_by_name(argv[i + 1]);
            if (!dt) {
                fprintf(stderr, "Unknown dtype: %s\nSupported:", argv[i + 1]);
                for (size_t k = 0; k < NUM_DTYPES; k++) fprintf(stderr, " %s", DTYPE_TABLE[k].name);
                fprintf(stderr, "\n");
                return 1;
            }
            g_dtype = dt;
            /* remove both --dtype and the value */
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2; i--;
        }
    }

    if (argc < 3 || argc > 7) {
        fprintf(stderr, "Usage: %s [--dtype <type>] <matrix.mtx|-> <out_name> [block_rows] [max_rows] [row_offset] [subdir]\n", argv[0]);
        fprintf(stderr, "  <matrix.mtx|->: MatrixMarket file path, or '-' to stream from stdin\n");
        fprintf(stderr, "  --binary: COO payload after the size line is raw binary (SoA: int32 rows,\n");
        fprintf(stderr, "            int32 cols, <dtype> vals; all 0-based). Requires input '-'.\n");
        fprintf(stderr, "  --level <1-22>: zstd compression level (default: 1).\n");
        fprintf(stderr, "  --dtype <type>: Data type (default: uint16). Supported:");
        for (size_t k = 0; k < NUM_DTYPES; k++) fprintf(stderr, " %s", DTYPE_TABLE[k].name);
        fprintf(stderr, "\n  --float32: Shorthand for --dtype float32\n");
        fprintf(stderr, "  Default: block_rows=16, max_rows=8192, row_offset=0, subdir=X_RM\n");
        return 1;
    }

    printf("Mode: %s (version %u, %zu bytes/value), zstd level %d\n",
           g_dtype->name, g_dtype->version, g_dtype->val_size, g_zstd_level);

    const char *mtx_path = argv[1];
    const char *out_name = argv[2];
    uint32_t block_rows = DEFAULT_BLOCK_ROWS;
    uint32_t max_rows = DEFAULT_MAX_ROWS;
    long long row_offset = 0;
    const char *subdir = "X_RM";

    if (argc >= 4) {
        block_rows = (uint32_t)atoi(argv[3]);
        if (block_rows == 0 || block_rows > 256) {
            fprintf(stderr, "Error: block_rows must be 1–256, got %u\n", block_rows);
            return 1;
        }
    }
    if (argc >= 5) {
        max_rows = (uint32_t)atoi(argv[4]);
        if (max_rows == 0 || max_rows > 1000000) {
            fprintf(stderr, "Error: max_rows must be 1–1000000, got %u\n", max_rows);
            return 1;
        }
    }
    if (argc >= 6) { row_offset = atoll(argv[5]); if (row_offset < 0) { fprintf(stderr, "row_offset must be >= 0\n"); return 1; } }
    if (argc >= 7) { subdir = argv[6]; }

    /* "-" means read the MatrixMarket stream from stdin (no intermediate
       file on disk).  Otherwise open the named file. */
    int from_stdin = (strcmp(mtx_path, "-") == 0);
    FILE *f = from_stdin ? stdin : fopen(mtx_path, "r");
    if (!f) { perror("open mtx"); return 1; }
    setvbuf(f, NULL, _IOFBF, 1024 * 1024);

    char line[4096];
    if (!fgets(line, sizeof(line), f)) { fprintf(stderr, "Failed to read header\n"); fclose(f); return 1; }
    if (strncmp(line, "%%MatrixMarket", 14) != 0) { fprintf(stderr, "Not a MatrixMarket file\n"); fclose(f); return 1; }
    if (!skip_to_size_line(f, line, sizeof(line))) { fprintf(stderr, "Unexpected EOF\n"); fclose(f); return 1; }

    long long nrows_ll, ncols_ll, nnz_total_ll;
    if (sscanf(line, "%lld %lld %lld", &nrows_ll, &ncols_ll, &nnz_total_ll) != 3) {
        fprintf(stderr, "Failed to parse dimensions\n"); fclose(f); return 1;
    }
    if (ncols_ll > UINT32_MAX) { fprintf(stderr, "ncols too large\n"); fclose(f); return 1; }
    const uint32_t ncols = (uint32_t)ncols_ll;

    printf("Matrix: %lld rows, %lld cols, %lld nnz (global)\n", nrows_ll, ncols_ll, nnz_total_ll);
    printf("Processing in chunks of %u rows, %u rows per block\n", max_rows, block_rows);
    fflush(stdout);

    /* Create output directories */
    char out_dir[1024];
    snprintf(out_dir, sizeof(out_dir), "%s", out_name);
    if (mkdir(out_dir, 0755) != 0 && errno != EEXIST) { perror("mkdir"); fclose(f); return 1; }

    char chunk_dir[2048];
    int n = snprintf(chunk_dir, sizeof(chunk_dir), "%s/%s", out_dir, subdir);
    if (n < 0 || n >= (int)sizeof(chunk_dir)) { fprintf(stderr, "Path too long\n"); fclose(f); return 1; }
    if (mkdir(chunk_dir, 0755) != 0 && errno != EEXIST) { perror("mkdir subdir"); fclose(f); return 1; }
    printf("Output %s directory: %s\n", subdir, chunk_dir);
    printf("Output directory: %s\n", out_dir);
    fflush(stdout);

    /* Build CSR and process chunks */
    printf("Building CSR structure from MTX file...\n");
    fflush(stdout);
    FullCSR *csr = g_binary_input
        ? build_full_csr_binary(f, nrows_ll, ncols, nnz_total_ll)
        : build_full_csr(f, nrows_ll, ncols);
    if (!csr) { if (!from_stdin) fclose(f); return 1; }
    if (!from_stdin) fclose(f);

    int chunk_num = 0;
    for (long long sr = 0; sr < nrows_ll; sr += max_rows) {
        long long er = sr + max_rows;
        if (er > nrows_ll) er = nrows_ll;

        char out_path[4096];
        int pn = snprintf(out_path, sizeof(out_path), "%s/%s/%d.bin", out_dir, subdir, chunk_num);
        if (pn < 0 || pn >= (int)sizeof(out_path)) { fprintf(stderr, "Path too long\n"); free_full_csr(csr); return 1; }

        if (!process_chunk_from_csr(csr, sr + row_offset, er + row_offset, out_path,
                                    block_rows, max_rows, row_offset, ncols)) {
            fprintf(stderr, "Failed chunk %d\n", chunk_num);
            free_full_csr(csr); return 1;
        }
        chunk_num++;
    }

    free_full_csr(csr);
    printf("\nAll chunks processed successfully!\n");
    fflush(stdout);
    return 0;
}
