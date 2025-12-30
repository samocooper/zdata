#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "zstd.h"
#include "zstd_seekable.h"   /* from contrib/seekable_format */

#define BLOCK_ROWS 16
#define MAX_ROWS   4096

/* Simple block-CSR structure (for 16 rows) */
typedef struct {
    int32_t nnz;         /* total nnz in this block */
    int32_t *indptr;     /* length BLOCK_ROWS+1 */
    int32_t *indices;    /* length nnz */
    float   *data;       /* length nnz */
    int32_t write_pos[BLOCK_ROWS];
} BlockCSR;

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

static void write_lef32(uint8_t *dst, float f) {
    uint32_t v;
    memcpy(&v, &f, sizeof(v));
    write_le32(dst, v);
}

/* Serialize a 16-row CSR block into a single contiguous blob.
   Layout (all little-endian):
     u32 magic ('ZCSR' = 0x5253435A)
     u32 version (1)
     u32 start_row
     u32 nrows_in_block (always 16 except last, but here always 16 for first 4096)
     u32 ncols
     u32 nnz
     i32 indptr[17]
     i32 indices[nnz]
     f32 data[nnz]
*/
static uint8_t* serialize_block(
    const BlockCSR *b,
    int32_t start_row,
    int32_t nrows_in_block,
    int32_t ncols,
    size_t *out_size
) {
    const uint32_t magic = 0x5253435A; /* 'ZCSR' little-endian bytes */
    const uint32_t version = 1;

    size_t header_bytes = 6 * 4; /* 6 u32 */
    size_t indptr_bytes = (BLOCK_ROWS + 1) * sizeof(int32_t);
    size_t indices_bytes = (size_t)b->nnz * sizeof(int32_t);
    size_t data_bytes = (size_t)b->nnz * sizeof(float);

    size_t total = header_bytes + indptr_bytes + indices_bytes + data_bytes;
    uint8_t *buf = (uint8_t*)malloc(total);
    if (!buf) return NULL;

    size_t off = 0;
    write_le32(buf + off, magic); off += 4;
    write_le32(buf + off, version); off += 4;
    write_le32(buf + off, (uint32_t)start_row); off += 4;
    write_le32(buf + off, (uint32_t)nrows_in_block); off += 4;
    write_le32(buf + off, (uint32_t)ncols); off += 4;
    write_le32(buf + off, (uint32_t)b->nnz); off += 4;

    /* indptr */
    for (int i = 0; i < BLOCK_ROWS + 1; i++) {
        write_le32(buf + off, (uint32_t)b->indptr[i]);
        off += 4;
    }

    /* indices */
    for (int32_t i = 0; i < b->nnz; i++) {
        write_le32(buf + off, (uint32_t)b->indices[i]);
        off += 4;
    }

    /* data */
    for (int32_t i = 0; i < b->nnz; i++) {
        write_lef32(buf + off, b->data[i]);
        off += 4;
    }

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

/* Count nnz per row for all rows in the file (cached for reuse) */
static int32_t* count_all_rows(FILE *f, long data_start_pos, long long nrows) {
    int32_t *row_counts = (int32_t*)calloc((size_t)nrows, sizeof(int32_t));
    if (!row_counts) {
        fprintf(stderr, "OOM row_counts\n");
        return NULL;
    }

    char line[4096];
    long long row, col;
    double value;

    fseek(f, data_start_pos, SEEK_SET);
    while (fgets(line, sizeof(line), f)) {
        int n = sscanf(line, "%lld %lld %lf", &row, &col, &value);
        if (n < 2) continue;
        row -= 1;
        if (row >= 0 && row < nrows) {
            row_counts[row] += 1;
        }
    }

    return row_counts;
}

/* Process a chunk of rows (start_row to end_row-1) and write to output file */
static int process_chunk(
    FILE *f,
    long data_start_pos,
    int32_t ncols,
    long long start_row_global,
    long long end_row_global,
    const int32_t *cached_row_counts,  /* Pre-computed counts for all rows */
    const char *out_path
) {
    int32_t chunk_size = (int32_t)(end_row_global - start_row_global);
    if (chunk_size <= 0) return 1;
    if (chunk_size > MAX_ROWS) chunk_size = MAX_ROWS;

    /* Extract row counts for this chunk from cache */
    int32_t *row_counts = (int32_t*)malloc(chunk_size * sizeof(int32_t));
    if (!row_counts) { fprintf(stderr, "OOM row_counts\n"); return 0; }

    for (int32_t i = 0; i < chunk_size; i++) {
        row_counts[i] = cached_row_counts[start_row_global + i];
    }

    char line[4096];
    long long row, col;
    double value;

    /* Allocate blocks */
    int32_t nblocks = (chunk_size + BLOCK_ROWS - 1) / BLOCK_ROWS; /* ceil division */
    BlockCSR *blocks = (BlockCSR*)calloc(nblocks, sizeof(BlockCSR));
    if (!blocks) { fprintf(stderr, "OOM blocks\n"); free(row_counts); return 0; }

    for (int32_t b = 0; b < nblocks; b++) {
        blocks[b].indptr = (int32_t*)malloc((BLOCK_ROWS + 1) * sizeof(int32_t));
        if (!blocks[b].indptr) { fprintf(stderr, "OOM indptr\n"); return 0; }

        blocks[b].indptr[0] = 0;
        for (int32_t r = 0; r < BLOCK_ROWS; r++) {
            int32_t local_row = b * BLOCK_ROWS + r;
            int32_t c = (local_row < chunk_size) ? row_counts[local_row] : 0;
            blocks[b].indptr[r + 1] = blocks[b].indptr[r] + c;
        }

        blocks[b].nnz = blocks[b].indptr[BLOCK_ROWS];
        if (blocks[b].nnz > 0) {
            blocks[b].indices = (int32_t*)malloc((size_t)blocks[b].nnz * sizeof(int32_t));
            blocks[b].data    = (float*)malloc((size_t)blocks[b].nnz * sizeof(float));
            if (!blocks[b].indices || !blocks[b].data) {
                fprintf(stderr, "OOM indices/data\n");
                return 0;
            }
        }

        for (int32_t r = 0; r < BLOCK_ROWS; r++) {
            blocks[b].write_pos[r] = blocks[b].indptr[r];
        }
    }

    /* Pass 2: fill blocks */
    fseek(f, data_start_pos, SEEK_SET);
    while (fgets(line, sizeof(line), f)) {
        int n = sscanf(line, "%lld %lld %lf", &row, &col, &value);
        if (n < 2) continue;

        row -= 1;
        col -= 1;
        if (row < start_row_global || row >= end_row_global) continue;

        int32_t local_row = (int32_t)(row - start_row_global);
        int32_t b = local_row / BLOCK_ROWS;
        int32_t r = local_row % BLOCK_ROWS;
        int32_t pos = blocks[b].write_pos[r]++;

        blocks[b].indices[pos] = (int32_t)col;
        blocks[b].data[pos]    = (n == 3) ? (float)value : 1.0f;
    }

    /* Write seekable archive: 1 frame per block */
    FILE *out = fopen(out_path, "wb");
    if (!out) { perror("open out"); return 0; }

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

    for (int32_t b = 0; b < nblocks; b++) {
        int32_t start_row = (int32_t)start_row_global + b * BLOCK_ROWS;
        int32_t nrows_in_block = BLOCK_ROWS;
        if (b == nblocks - 1) {
            /* Last block may have fewer rows */
            int32_t remaining = chunk_size - b * BLOCK_ROWS;
            if (remaining < BLOCK_ROWS) nrows_in_block = remaining;
        }

        size_t blob_sz = 0;
        uint8_t *blob = serialize_block(&blocks[b], start_row, nrows_in_block, ncols, &blob_sz);
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

        if ((b % 32) == 0) {
            printf("  Wrote frame for block %d/%d\n", b, nblocks);
        }
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
    for (int32_t b = 0; b < nblocks; b++) {
        free(blocks[b].indptr);
        free(blocks[b].indices);
        free(blocks[b].data);
    }
    free(blocks);
    free(row_counts);

    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix.mtx> <out_name>\n", argv[0]);
        fprintf(stderr, "  Example: %s matrix.mtx andrews\n", argv[0]);
        fprintf(stderr, "  Output: andrews.zdata/0.bin, andrews.zdata/1.bin, ...\n");
        return 1;
    }

    const char *mtx_path = argv[1];
    const char *out_name = argv[2];

    FILE *f = fopen(mtx_path, "r");
    if (!f) { perror("open mtx"); return 1; }

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
    if (ncols_ll > INT32_MAX) {
        fprintf(stderr, "ncols too large for int32\n");
        fclose(f);
        return 1;
    }
    const int32_t ncols = (int32_t)ncols_ll;

    long data_start_pos = ftell(f);

    printf("Matrix: %lld rows, %lld cols, %lld nnz (global)\n", nrows_ll, ncols_ll, nnz_total_ll);
    printf("Processing in chunks of %d rows\n", MAX_ROWS);

    /* Create output directory */
    char out_dir[1024];
    snprintf(out_dir, sizeof(out_dir), "%s.zdata", out_name);
    
    if (mkdir(out_dir, 0755) != 0) {
        if (errno != EEXIST) {
            perror("mkdir");
            fclose(f);
            return 1;
        }
        /* Directory already exists, that's okay */
    }
    printf("Output directory: %s\n", out_dir);

    /* Cache: count nnz per row for all rows (single pass) */
    printf("Counting non-zeros per row (caching for all chunks)...\n");
    int32_t *cached_row_counts = count_all_rows(f, data_start_pos, nrows_ll);
    if (!cached_row_counts) {
        fclose(f);
        return 1;
    }
    printf("Cached row counts complete.\n");

    /* Process file in chunks of MAX_ROWS */
    int chunk_num = 0;  /* Start from 0 instead of 1 */
    for (long long start_row = 0; start_row < nrows_ll; start_row += MAX_ROWS) {
        long long end_row = start_row + MAX_ROWS;
        if (end_row > nrows_ll) end_row = nrows_ll;

        char out_path[4096];  /* Increased size to handle long paths and avoid truncation warning */
        int n = snprintf(out_path, sizeof(out_path), "%s/%d.bin", out_dir, chunk_num);
        if (n < 0 || n >= (int)sizeof(out_path)) {
            fprintf(stderr, "ERROR: Path too long for output file\n");
            free(cached_row_counts);
            fclose(f);
            return 1;
        }

        printf("\nProcessing chunk %d: rows %lld-%lld -> %s\n",
               chunk_num, start_row, end_row - 1, out_path);

        if (!process_chunk(f, data_start_pos, ncols, start_row, end_row, cached_row_counts, out_path)) {
            fprintf(stderr, "Failed to process chunk %d\n", chunk_num);
            free(cached_row_counts);
            fclose(f);
            return 1;
        }

        printf("Done: wrote %s\n", out_path);
        chunk_num++;
    }

    free(cached_row_counts);
    fclose(f);
    printf("\nAll chunks processed successfully!\n");
    return 0;
}
