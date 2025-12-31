#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "zstd.h"
#include "zstd_seekable.h"   /* from contrib/seekable_format */

/* Default values (can be overridden via command line) */
#define DEFAULT_BLOCK_ROWS 16
#define DEFAULT_MAX_ROWS   8192

/* Simple block-CSR structure (variable number of rows per block) */
typedef struct {
    uint32_t nnz;         /* total nnz in this block */
    uint32_t *indptr;     /* length block_rows+1 */
    uint32_t *indices;    /* length nnz */
    uint16_t *data;       /* length nnz */
    uint32_t *write_pos;  /* length block_rows (allocated dynamically) */
} BlockCSR;

/* Full CSR structure for the entire matrix */
typedef struct {
    uint32_t *indptr;     /* length nrows+1 */
    uint32_t *indices;    /* length nnz_total */
    uint16_t *data;       /* length nnz_total */
    long long nnz_total;  /* total non-zeros */
} FullCSR;

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


/* Serialize a CSR block into a single contiguous blob.
   Layout (all little-endian):
     u32 magic ('ZCSR' = 0x5253435A)
     u32 version (2)
     u32 start_row
     u32 nrows_in_block
     u32 ncols
     u32 nnz
     u32 indptr[nrows_in_block+1]
     u32 indices[nnz]
     u16 data[nnz]
*/
static uint8_t* serialize_block(
    const BlockCSR *b,
    uint32_t start_row,
    uint32_t nrows_in_block,
    uint32_t ncols,
    uint32_t block_rows,  /* Maximum rows per block (for indptr size) */
    size_t *out_size
) {
    const uint32_t magic = 0x5253435A; /* 'ZCSR' little-endian bytes */
    const uint32_t version = 2;  /* Version 2: uint16_t data, uint32_t indices/indptr */

    size_t header_bytes = 6 * 4; /* 6 u32 */
    size_t indptr_bytes = (block_rows + 1) * sizeof(uint32_t);
    size_t indices_bytes = (size_t)b->nnz * sizeof(uint32_t);
    size_t data_bytes = (size_t)b->nnz * sizeof(uint16_t);

    size_t total = header_bytes + indptr_bytes + indices_bytes + data_bytes;
    uint8_t *buf = (uint8_t*)malloc(total);
    if (!buf) return NULL;

    size_t off = 0;
    write_le32(buf + off, magic); off += 4;
    write_le32(buf + off, version); off += 4;
    write_le32(buf + off, start_row); off += 4;
    write_le32(buf + off, nrows_in_block); off += 4;
    write_le32(buf + off, ncols); off += 4;
    write_le32(buf + off, (uint32_t)b->nnz); off += 4;

    /* indptr - copy directly (data is already in memory, assumes little-endian) */
    memcpy(buf + off, b->indptr, indptr_bytes);
    off += indptr_bytes;

    /* indices - copy directly (data is already in memory, assumes little-endian) */
    memcpy(buf + off, b->indices, indices_bytes);
    off += indices_bytes;

    /* data - copy directly (data is already in memory, assumes little-endian) */
    memcpy(buf + off, b->data, data_bytes);
    off += data_bytes;

    *out_size = total;
    return buf;
}

static int flush_seekable_to_file(
    ZSTD_seekable_CStream *zcs,
    FILE *out,
    const void *src,
    size_t srcSize
) {
    /* Stream input -> output file */
    ZSTD_inBuffer inb = { src, srcSize, 0 };
    uint8_t outbuf[1 << 20]; /* 1 MiB staging buffer */

    while (inb.pos < inb.size) {
        ZSTD_outBuffer outb = { outbuf, sizeof(outbuf), 0 };
        size_t r = ZSTD_seekable_compressStream(zcs, &outb, &inb);
        if (ZSTD_isError(r)) {
            fprintf(stderr, "ZSTD_seekable_compressStream error: %s\n", ZSTD_getErrorName(r));
            return 0;
        }
        if (outb.pos) {
            if (fwrite(outbuf, 1, outb.pos, out) != outb.pos) {
                perror("fwrite");
                return 0;
            }
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
            fprintf(stderr, "ZSTD_seekable_endFrame error: %s\n", ZSTD_getErrorName(r));
            return 0;
        }
        if (outb.pos) {
            if (fwrite(outbuf, 1, outb.pos, out) != outb.pos) {
                perror("fwrite");
                return 0;
            }
        }
        if (r == 0) break; /* frame ended & flushed */
    }
    return 1;
}

static int end_stream_to_file(ZSTD_seekable_CStream *zcs, FILE *out) {
    uint8_t outbuf[1 << 20];
    for (;;) {
        ZSTD_outBuffer outb = { outbuf, sizeof(outbuf), 0 };
        size_t r = ZSTD_seekable_endStream(zcs, &outb);
        if (ZSTD_isError(r)) {
            fprintf(stderr, "ZSTD_seekable_endStream error: %s\n", ZSTD_getErrorName(r));
            return 0;
        }
        if (outb.pos) {
            if (fwrite(outbuf, 1, outb.pos, out) != outb.pos) {
                perror("fwrite");
                return 0;
            }
        }
        if (r == 0) break; /* seek table fully flushed */
    }
    return 1;
}

/* Dynamic array structure for collecting entries per row */
typedef struct {
    uint32_t *indices;
    uint16_t *data;
    uint32_t size;
    uint32_t capacity;
} RowEntries;

/* Fast manual parsing - much faster than sscanf */
static int parse_line_fast(const char *line, long long *row, long long *col, double *value) {
    const char *p = line;
    
    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n') return 0;
    
    /* Parse row */
    *row = 0;
    while (*p >= '0' && *p <= '9') {
        *row = *row * 10 + (*p - '0');
        p++;
    }
    
    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n') return 1;  /* Only row and col */
    
    /* Parse col */
    *col = 0;
    while (*p >= '0' && *p <= '9') {
        *col = *col * 10 + (*p - '0');
        p++;
    }
    
    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n') return 2;  /* Row and col only */
    
    /* Parse value (if present) */
    *value = 0.0;
    int sign = 1;
    if (*p == '-') {
        sign = -1;
        p++;
    } else if (*p == '+') {
        p++;
    }
    
    /* Integer part */
    while (*p >= '0' && *p <= '9') {
        *value = *value * 10.0 + (*p - '0');
        p++;
    }
    
    /* Decimal part */
    if (*p == '.') {
        p++;
        double frac = 0.1;
        while (*p >= '0' && *p <= '9') {
            *value += (*p - '0') * frac;
            frac *= 0.1;
            p++;
        }
    }
    
    /* Scientific notation (basic support) */
    if (*p == 'e' || *p == 'E') {
        p++;
        int exp_sign = 1;
        if (*p == '-') {
            exp_sign = -1;
            p++;
        } else if (*p == '+') {
            p++;
        }
        int exp = 0;
        while (*p >= '0' && *p <= '9') {
            exp = exp * 10 + (*p - '0');
            p++;
        }
        for (int i = 0; i < exp; i++) {
            if (exp_sign > 0) *value *= 10.0;
            else *value *= 0.1;
        }
    }
    
    *value *= sign;
    return 3;  /* Row, col, and value */
}

/* Build full CSR structure from MTX file (single pass - optimized parsing) */
static FullCSR* build_full_csr(FILE *f, long data_start_pos, long long nrows, long long ncols) {
    /* Set input file to fully buffered for better performance */
    setvbuf(f, NULL, _IOFBF, 1024 * 1024);  /* 1MB buffer */
    
    /* Allocate dynamic arrays for each row */
    RowEntries *rows = (RowEntries*)calloc((size_t)nrows, sizeof(RowEntries));
    if (!rows) {
        fprintf(stderr, "OOM rows\n");
        return NULL;
    }
    
    /* Estimate initial capacity based on average nnz per row (if we have nnz info) */
    /* For now, use a reasonable default that reduces reallocations */
    const uint32_t INITIAL_CAPACITY = 64;  /* Increased from 16 to reduce reallocations */
    for (long long i = 0; i < nrows; i++) {
        rows[i].capacity = INITIAL_CAPACITY;
        rows[i].indices = (uint32_t*)malloc(INITIAL_CAPACITY * sizeof(uint32_t));
        rows[i].data = (uint16_t*)malloc(INITIAL_CAPACITY * sizeof(uint16_t));
        if (!rows[i].indices || !rows[i].data) {
            fprintf(stderr, "OOM initial row arrays\n");
            /* Cleanup */
            for (long long j = 0; j <= i; j++) {
                if (rows[j].indices) free(rows[j].indices);
                if (rows[j].data) free(rows[j].data);
            }
            free(rows);
            return NULL;
        }
        rows[i].size = 0;
    }

    char line[8192];  /* Increased buffer size for longer lines */
    long long row, col;
    double value = 0.0;  /* Initialize to avoid warning */
    long long lines_processed = 0;
    long long last_progress = 0;

    printf("  Building CSR structure (single pass, optimized)...\n");
    fflush(stdout);
    fseek(f, data_start_pos, SEEK_SET);
    
    while (fgets(line, sizeof(line), f)) {
        /* Fast manual parsing instead of sscanf */
        int n = parse_line_fast(line, &row, &col, &value);
        if (n < 2) continue;
        
        /* Progress reporting every 50 million lines (reduced frequency for less I/O overhead) */
        lines_processed++;
        if (lines_processed - last_progress >= 50000000) {
            printf("    Processed %lld lines...\n", lines_processed);
            fflush(stdout);
            last_progress = lines_processed;
        }
        
        /* Convert from 1-based MTX to 0-based local row index */
        long long local_row = row - 1;
        col -= 1;
        
        /* Validate local row and column indices (before applying offset) */
        if (local_row < 0 || local_row >= nrows || col < 0 || col >= ncols) continue;
        
        RowEntries *r = &rows[local_row];
        
        /* Grow array if needed (exponential growth) */
        if (r->size >= r->capacity) {
            uint32_t new_capacity = r->capacity * 2;
            uint32_t *new_indices = (uint32_t*)realloc(r->indices, new_capacity * sizeof(uint32_t));
            uint16_t *new_data = (uint16_t*)realloc(r->data, new_capacity * sizeof(uint16_t));
            if (!new_indices || !new_data) {
                fprintf(stderr, "OOM realloc row arrays\n");
                /* Cleanup */
                for (long long i = 0; i < nrows; i++) {
                    free(rows[i].indices);
                    free(rows[i].data);
                }
                free(rows);
                return NULL;
            }
            r->indices = new_indices;
            r->data = new_data;
            r->capacity = new_capacity;
        }
        
        /* Add entry */
        r->indices[r->size] = (uint32_t)col;
        
        /* Convert value to uint16_t with clamping */
        if (n == 3) {
            double clamped = value;
            if (clamped < 0.0) clamped = 0.0;
            if (clamped > (double)UINT16_MAX) clamped = (double)UINT16_MAX;
            r->data[r->size] = (uint16_t)(clamped + 0.5);
        } else {
            r->data[r->size] = 1;  /* Default value when no value field */
        }
        r->size++;
    }
    
    if (lines_processed > 0 && lines_processed >= 50000000) {
        printf("    Completed reading: %lld total lines processed\n", lines_processed);
        fflush(stdout);
    }

    /* Build indptr and compute total nnz */
    printf("  Converting to CSR format...\n");
    fflush(stdout);
    uint32_t *indptr = (uint32_t*)malloc((size_t)(nrows + 1) * sizeof(uint32_t));
    if (!indptr) {
        fprintf(stderr, "OOM indptr\n");
        for (long long i = 0; i < nrows; i++) {
            free(rows[i].indices);
            free(rows[i].data);
        }
        free(rows);
        return NULL;
    }
    
    indptr[0] = 0;
    long long nnz_total = 0;
    for (long long i = 0; i < nrows; i++) {
        nnz_total += rows[i].size;
        indptr[i + 1] = (uint32_t)nnz_total;
    }
    
    if (nnz_total == 0) {
        fprintf(stderr, "Error: matrix has no non-zero elements\n");
        free(indptr);
        for (long long i = 0; i < nrows; i++) {
            free(rows[i].indices);
            free(rows[i].data);
        }
        free(rows);
        return NULL;
    }

    /* Allocate final CSR arrays */
    uint32_t *indices = (uint32_t*)malloc((size_t)nnz_total * sizeof(uint32_t));
    uint16_t *data = (uint16_t*)malloc((size_t)nnz_total * sizeof(uint16_t));
    if (!indices || !data) {
        fprintf(stderr, "OOM indices/data\n");
        free(indptr);
        for (long long i = 0; i < nrows; i++) {
            free(rows[i].indices);
            free(rows[i].data);
        }
        free(rows);
        if (indices) free(indices);
        if (data) free(data);
        return NULL;
    }

    /* Copy data from row arrays to CSR arrays */
    uint32_t pos = 0;
    for (long long i = 0; i < nrows; i++) {
        if (rows[i].size > 0) {
            memcpy(indices + pos, rows[i].indices, rows[i].size * sizeof(uint32_t));
            memcpy(data + pos, rows[i].data, rows[i].size * sizeof(uint16_t));
            pos += rows[i].size;
        }
        /* Free row arrays as we go */
        free(rows[i].indices);
        free(rows[i].data);
    }
    free(rows);
    
    FullCSR *csr = (FullCSR*)malloc(sizeof(FullCSR));
    if (!csr) {
        fprintf(stderr, "OOM FullCSR\n");
        free(indptr);
        free(indices);
        free(data);
        return NULL;
    }
    
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
        if (csr->indptr) free(csr->indptr);
        if (csr->indices) free(csr->indices);
        if (csr->data) free(csr->data);
        free(csr);
    }
}

/* Process a chunk of rows from CSR structure (faster - no file re-reading) */
static int process_chunk_from_csr(
    const FullCSR *csr,
    long long start_row_global,
    long long end_row_global,
    const char *out_path,
    uint32_t block_rows,
    uint32_t max_rows,
    long long row_offset,
    uint32_t ncols  /* ncols from MTX file header - ensures consistency across all chunks */
) {
    uint32_t chunk_size = (uint32_t)(end_row_global - start_row_global);
    if (chunk_size == 0) return 1;
    if (chunk_size > max_rows) chunk_size = max_rows;

    long long start_row_local = start_row_global - row_offset;
    
    /* Extract row counts from indptr: indptr[i+1] - indptr[i] = nnz in row i */
    uint32_t *row_counts = (uint32_t*)malloc(chunk_size * sizeof(uint32_t));
    if (!row_counts) { fprintf(stderr, "OOM row_counts\n"); return 0; }

    for (uint32_t i = 0; i < chunk_size; i++) {
        long long row_idx = start_row_local + i;
        row_counts[i] = csr->indptr[row_idx + 1] - csr->indptr[row_idx];
    }

    /* Allocate blocks */
    uint32_t nblocks = (chunk_size + block_rows - 1) / block_rows;
    BlockCSR *blocks = (BlockCSR*)calloc(nblocks, sizeof(BlockCSR));
    if (!blocks) { fprintf(stderr, "OOM blocks\n"); free(row_counts); return 0; }

    for (uint32_t b = 0; b < nblocks; b++) {
        blocks[b].indptr = (uint32_t*)malloc((block_rows + 1) * sizeof(uint32_t));
        if (!blocks[b].indptr) { fprintf(stderr, "OOM indptr\n"); return 0; }

        blocks[b].write_pos = (uint32_t*)malloc(block_rows * sizeof(uint32_t));
        if (!blocks[b].write_pos) { fprintf(stderr, "OOM write_pos\n"); return 0; }

        blocks[b].indptr[0] = 0;
        for (uint32_t r = 0; r < block_rows; r++) {
            uint32_t local_row = b * block_rows + r;
            blocks[b].indptr[r + 1] = blocks[b].indptr[r] + 
                ((local_row < chunk_size) ? row_counts[local_row] : 0);
        }

        blocks[b].nnz = blocks[b].indptr[block_rows];
        if (blocks[b].nnz > 0) {
            blocks[b].indices = (uint32_t*)malloc((size_t)blocks[b].nnz * sizeof(uint32_t));
            blocks[b].data    = (uint16_t*)malloc((size_t)blocks[b].nnz * sizeof(uint16_t));
            if (!blocks[b].indices || !blocks[b].data) {
                fprintf(stderr, "OOM indices/data\n");
                return 0;
            }
        }

        for (uint32_t r = 0; r < block_rows; r++) {
            blocks[b].write_pos[r] = blocks[b].indptr[r];
        }
    }

    /* Fill blocks with data from CSR structure (much faster - no file I/O) */
    for (uint32_t i = 0; i < chunk_size; i++) {
        long long row_idx = start_row_local + i;
        uint32_t row_start = csr->indptr[row_idx];
        uint32_t row_end = csr->indptr[row_idx + 1];
        uint32_t row_nnz = row_end - row_start;
        
        uint32_t b = i / block_rows;
        uint32_t r = i % block_rows;
        
        /* Copy indices and data for this row from CSR */
        for (uint32_t j = 0; j < row_nnz; j++) {
            uint32_t pos = blocks[b].write_pos[r]++;
            blocks[b].indices[pos] = csr->indices[row_start + j];
            blocks[b].data[pos] = csr->data[row_start + j];
        }
    }

    /* Write seekable archive: 1 frame per block */
    FILE *out = fopen(out_path, "wb");
    if (!out) { perror("open out"); return 0; }
    
    /* Set output file to fully buffered for better performance (reduces system calls) */
    setvbuf(out, NULL, _IOFBF, 256 * 1024);  /* 256KB buffer */

    ZSTD_seekable_CStream *zcs = ZSTD_seekable_createCStream();
    if (!zcs) {
        fprintf(stderr, "ZSTD_seekable_createCStream failed\n");
        fclose(out);
        return 0;
    }

    int compressionLevel = 3;
    int checksumFlag = 0;
    unsigned maxFrameSize = 0;
    size_t initR = ZSTD_seekable_initCStream(zcs, compressionLevel, checksumFlag, maxFrameSize);
    if (ZSTD_isError(initR)) {
        fprintf(stderr, "ZSTD_seekable_initCStream error: %s\n", ZSTD_getErrorName(initR));
        ZSTD_seekable_freeCStream(zcs);
        fclose(out);
        return 0;
    }

    /* Use ncols from MTX file header (passed as parameter) */
    /* This ensures consistency across all chunks, even if some columns have no non-zeros */

    for (uint32_t b = 0; b < nblocks; b++) {
        uint32_t start_row = (uint32_t)start_row_global + b * block_rows;
        uint32_t nrows_in_block = (b == nblocks - 1) ? 
            (chunk_size - b * block_rows) : block_rows;

        size_t blob_sz = 0;
        uint8_t *blob = serialize_block(&blocks[b], start_row, nrows_in_block, ncols, block_rows, &blob_sz);
        if (!blob) {
            fprintf(stderr, "OOM serialize_block\n");
            ZSTD_seekable_freeCStream(zcs);
            fclose(out);
            return 0;
        }

        if (!flush_seekable_to_file(zcs, out, blob, blob_sz)) {
            free(blob);
            ZSTD_seekable_freeCStream(zcs);
            fclose(out);
            return 0;
        }
        free(blob);

        if (!end_frame_to_file(zcs, out)) {
            ZSTD_seekable_freeCStream(zcs);
            fclose(out);
            return 0;
        }

        /* Progress reporting removed for better performance */
    }

    /* Finish: writes the seek table */
    if (!end_stream_to_file(zcs, out)) {
        ZSTD_seekable_freeCStream(zcs);
        fclose(out);
        return 0;
    }

    ZSTD_seekable_freeCStream(zcs);
    fclose(out);

    /* Cleanup CSR blocks */
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
    if (argc < 3 || argc > 7) {
        fprintf(stderr, "Usage: %s <matrix.mtx> <out_name> [block_rows] [max_rows] [row_offset] [subdir]\n", argv[0]);
        fprintf(stderr, "  Example: %s matrix.mtx andrews\n", argv[0]);
        fprintf(stderr, "  Example: %s matrix.mtx andrews 16 8192\n", argv[0]);
        fprintf(stderr, "  Example: %s matrix.mtx andrews 16 8192 131072\n", argv[0]);
        fprintf(stderr, "  Example: %s matrix.mtx andrews 16 8192 0 X_CM\n", argv[0]);
        fprintf(stderr, "  Output: andrews/X_RM/0.bin, andrews/X_RM/1.bin, ... (or andrews/X_CM/0.bin, ...)\n");
        fprintf(stderr, "  Default: block_rows=16, max_rows=8192, row_offset=0, subdir=X_RM\n");
        fprintf(stderr, "  row_offset: Add this value to row indices from MTX file (for globally contiguous numbering)\n");
        fprintf(stderr, "  subdir: Subdirectory name for .bin files (default: X_RM, use X_CM for column-major)\n");
        return 1;
    }

    const char *mtx_path = argv[1];
    const char *out_name = argv[2];
    
    /* Parse optional block_rows, max_rows, row_offset, and subdir parameters */
    uint32_t block_rows = DEFAULT_BLOCK_ROWS;
    uint32_t max_rows = DEFAULT_MAX_ROWS;
    long long row_offset = 0;  /* Offset to add to row indices from MTX file */
    const char *subdir = "X_RM";  /* Default subdirectory name */
    
    if (argc >= 4) {
        block_rows = (uint32_t)atoi(argv[3]);
        if (block_rows == 0 || block_rows > 256) {
            fprintf(stderr, "Error: block_rows must be between 1 and 256, got %u\n", block_rows);
            return 1;
        }
    }
    
    if (argc >= 5) {
        max_rows = (uint32_t)atoi(argv[4]);
        if (max_rows == 0 || max_rows > 1000000) {
            fprintf(stderr, "Error: max_rows must be between 1 and 1000000, got %u\n", max_rows);
            return 1;
        }
    }
    
    if (argc >= 6) {
        row_offset = atoll(argv[5]);
        if (row_offset < 0) {
            fprintf(stderr, "Error: row_offset must be >= 0, got %lld\n", row_offset);
            return 1;
        }
    }
    
    if (argc >= 7) {
        subdir = argv[6];
    }

    FILE *f = fopen(mtx_path, "r");
    if (!f) { perror("open mtx"); return 1; }
    
    /* Set input file to fully buffered for better performance (reduces system calls) */
    /* Increased buffer size to 256KB for better read throughput */
    setvbuf(f, NULL, _IOFBF, 256 * 1024);  /* 256KB buffer */

    char line[4096];

    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Failed to read MatrixMarket header\n");
        fclose(f);
        return 1;
    }
    if (strncmp(line, "%%MatrixMarket", 14) != 0) {
        fprintf(stderr, "Not a MatrixMarket file\n");
        fclose(f);
        return 1;
    }

    if (!skip_to_size_line(f, line, sizeof(line))) {
        fprintf(stderr, "Unexpected EOF while reading header/comments\n");
        fclose(f);
        return 1;
    }

    long long nrows_ll, ncols_ll, nnz_total_ll;
    if (sscanf(line, "%lld %lld %lld", &nrows_ll, &ncols_ll, &nnz_total_ll) != 3) {
        fprintf(stderr, "Failed to parse matrix dimensions line\n");
        fclose(f);
        return 1;
    }
    if (ncols_ll > UINT32_MAX) {
        fprintf(stderr, "ncols too large for uint32\n");
        fclose(f);
        return 1;
    }
    const uint32_t ncols = (uint32_t)ncols_ll;

    long data_start_pos = ftell(f);

    printf("Matrix: %lld rows, %lld cols, %lld nnz (global)\n", nrows_ll, ncols_ll, nnz_total_ll);
    fflush(stdout);  /* Ensure output is flushed immediately */
    printf("Processing in chunks of %u rows, %u rows per block\n", max_rows, block_rows);
    fflush(stdout);  /* Ensure output is flushed immediately */

    /* Create output directory */
    char out_dir[1024];
    snprintf(out_dir, sizeof(out_dir), "%s", out_name);
    
    if (mkdir(out_dir, 0755) != 0) {
        if (errno != EEXIST) {
            perror("mkdir");
            fclose(f);
            return 1;
        }
        /* Directory already exists, that's okay */
    }
    
    /* Create subdirectory for .bin files (X_RM or X_CM) */
    char chunk_dir[2048];  /* Increased size to avoid truncation warning */
    int n = snprintf(chunk_dir, sizeof(chunk_dir), "%s/%s", out_dir, subdir);
    if (n < 0 || n >= (int)sizeof(chunk_dir)) {
        fprintf(stderr, "ERROR: Path too long for subdirectory\n");
        fclose(f);
        return 1;
    }
    if (mkdir(chunk_dir, 0755) != 0) {
        if (errno != EEXIST) {
            perror("mkdir subdirectory");
            fclose(f);
            return 1;
        }
        /* Directory already exists, that's okay */
    }
    printf("Output %s directory: %s\n", subdir, chunk_dir);
    fflush(stdout);
    
    printf("Output directory: %s\n", out_dir);
    fflush(stdout);  /* Ensure output is flushed immediately */

    /* Build full CSR structure once (uses indptr for row counts, eliminates file re-reading) */
    printf("Building CSR structure from MTX file...\n");
    fflush(stdout);  /* Ensure output is flushed immediately */
    FullCSR *csr = build_full_csr(f, data_start_pos, nrows_ll, ncols);
    if (!csr) {
        fclose(f);
        return 1;
    }
    
    /* We can close the file now - all data is in memory */
    fclose(f);

    /* Process file in chunks of max_rows */
    /* Adjust start_row and end_row by row_offset to account for global row numbering */
    int chunk_num = 0;  /* Start from 0 instead of 1 */
    for (long long start_row = 0; start_row < nrows_ll; start_row += max_rows) {
        long long end_row = start_row + max_rows;
        if (end_row > nrows_ll) end_row = nrows_ll;

        char out_path[4096];  /* Increased size to handle long paths and avoid truncation warning */
        int n = snprintf(out_path, sizeof(out_path), "%s/%s/%d.bin", out_dir, subdir, chunk_num);
        if (n < 0 || n >= (int)sizeof(out_path)) {
            fprintf(stderr, "ERROR: Path too long for output file\n");
            free_full_csr(csr);
            return 1;
        }

        long long global_start_row = start_row + row_offset;
        long long global_end_row = end_row + row_offset;
        
        if (!process_chunk_from_csr(csr, global_start_row, global_end_row, out_path, block_rows, max_rows, row_offset, ncols)) {
            fprintf(stderr, "Failed to process chunk %d\n", chunk_num);
            free_full_csr(csr);
            return 1;
        }

        chunk_num++;
    }

    free_full_csr(csr);
    printf("\nAll chunks processed successfully!\n");
    fflush(stdout);  /* Ensure output is flushed immediately */
    return 0;
}
