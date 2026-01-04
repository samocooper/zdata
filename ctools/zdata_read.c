#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "zstd.h"
#include "zstd_seekable.h"

#define MAX_BLOCK_ROWS 256  /* Maximum supported block_rows */
#define MAX_ROWS   1000000  /* Maximum supported max_rows */

static uint32_t read_le32(const uint8_t *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void write_le32(FILE *out, uint32_t v) {
    /* Optimize: write directly without temporary buffer */
    uint8_t b[4];
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
    fwrite(b, 1, 4, out);
}


typedef struct {
    uint32_t magic, version, start_row, nrows_in_block, ncols, nnz;
} BlockHeader;

static int parse_rows_csv(const char *s, uint32_t *out, int max_out) {
    int n = 0;
    const char *p = s;
    /* Optimize: single pass whitespace/comma skipping */
    while (*p) {
        /* Skip whitespace and commas in one pass */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') p++;
        if (!*p) break;
        
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p) break;
        if (n < max_out && v <= UINT32_MAX) {
            out[n++] = (uint32_t)v;
        }
        p = end;
    }
    return n;
}

static int parse_block(const uint8_t *buf, size_t sz,
                       BlockHeader *hdr,
                       const uint32_t **indptr,
                       const uint32_t **indices,
                       const uint16_t **data,
                       uint32_t *block_rows_out) {
    if (sz < 24 + 4) return 0;  /* Minimum header size */

    hdr->magic = read_le32(buf + 0);
    hdr->version = read_le32(buf + 4);
    hdr->start_row = read_le32(buf + 8);
    hdr->nrows_in_block = read_le32(buf + 12);
    hdr->ncols = read_le32(buf + 16);
    hdr->nnz = read_le32(buf + 20);

    if (hdr->magic != 0x5253435A || hdr->version != 2 || hdr->nrows_in_block == 0 || hdr->nrows_in_block > MAX_BLOCK_ROWS) {
        return 0;
    }

    /* Determine block_rows from indptr size: indptr is always block_rows+1 elements */
    /* We can infer block_rows by checking the size: (sz - 24 - indices - data) / 4 - 1 */
    /* But simpler: use nrows_in_block as a hint, but indptr is allocated for full block */
    /* Actually, we need to calculate: remaining bytes after header = sz - 24 */
    /* indptr_bytes = (block_rows + 1) * 4, so we can solve for block_rows */
    /* For now, we'll use a heuristic: check common sizes or infer from structure */
    
    /* Try to infer block_rows from the data structure */
    /* The indptr array size is (block_rows + 1) * 4 */
    /* After indptr comes indices (nnz * 4) and data (nnz * 2) */
    /* So: sz = 24 + (block_rows+1)*4 + nnz*4 + nnz*2 */
    /* Solving: (block_rows+1)*4 = sz - 24 - nnz*6 */
    /* block_rows = (sz - 24 - nnz*6) / 4 - 1 */
    
    size_t remaining = sz - 24;
    size_t data_indices_bytes = (size_t)hdr->nnz * (sizeof(uint32_t) + sizeof(uint16_t));
    if (remaining < data_indices_bytes) return 0;
    
    size_t indptr_bytes = remaining - data_indices_bytes;
    if (indptr_bytes % 4 != 0 || indptr_bytes < 4) return 0;
    
    uint32_t inferred_block_rows = (uint32_t)(indptr_bytes / 4) - 1;
    if (inferred_block_rows > MAX_BLOCK_ROWS || inferred_block_rows < hdr->nrows_in_block) {
        return 0;
    }
    
    uint32_t block_rows = inferred_block_rows;
    if (block_rows_out) *block_rows_out = block_rows;

    size_t off = 24;
    *indptr = (const uint32_t*)(buf + off);
    off += (block_rows + 1) * sizeof(uint32_t);

    /* Version 2: uint32_t indices, uint16_t data */
    size_t idx_bytes = (size_t)hdr->nnz * sizeof(uint32_t);
    size_t data_bytes = (size_t)hdr->nnz * sizeof(uint16_t);
    if (off + idx_bytes + data_bytes > sz) return 0;

    *indices = (const uint32_t*)(buf + off);
    off += idx_bytes;
    *data = (const uint16_t*)(buf + off);
    return 1;
}

static int decompress_frame(ZSTD_seekable *zs, unsigned frameIndex,
                            uint8_t **outBuf, size_t *outSize, size_t *bufCapacity) {
    size_t dsz = ZSTD_seekable_getFrameDecompressedSize(zs, frameIndex);
    if (ZSTD_isError(dsz)) {
        fprintf(stderr, "getFrameDecompressedSize error: %s\n", ZSTD_getErrorName(dsz));
        return 0;
    }

    /* Reuse buffer if it's large enough, otherwise reallocate */
    if (*outBuf == NULL || *bufCapacity < dsz) {
        if (*outBuf) free(*outBuf);
        *outBuf = (uint8_t*)malloc(dsz);
        if (!*outBuf) {
            fprintf(stderr, "OOM frame buffer (%zu)\n", dsz);
            return 0;
        }
        *bufCapacity = dsz;
    }

    size_t dr = ZSTD_seekable_decompressFrame(zs, *outBuf, dsz, frameIndex);
    if (ZSTD_isError(dr)) {
        fprintf(stderr, "decompressFrame error: %s\n", ZSTD_getErrorName(dr));
        return 0;
    }

    *outSize = dr;
    return 1;
}

/* Forward declaration */
static int process_file(const char *path, const char *rows_csv, uint32_t block_rows_override, int binary, int is_stdin_mode);

int main(int argc, char *argv[]) {
    int binary = 0;
    int argi = 1;
    uint32_t block_rows_override = 0;  /* 0 means auto-detect */
    int read_from_stdin = 0;  /* If true, read commands from stdin */

    /* Check if we should read from stdin (for persistent pool mode) */
    if (argc == 2 && strcmp(argv[1], "--binary") == 0) {
        binary = 1;
        read_from_stdin = 1;
    } else if (argc == 1) {
        read_from_stdin = 1;
    }

    if (read_from_stdin) {
        /* Persistent pool mode: read commands from stdin */
        char line[4096];
        while (fgets(line, sizeof(line), stdin) != NULL) {
            /* Parse command: "file_path\nrows_csv\nblock_rows\n" */
            char *path = line;
            /* Remove newline */
            size_t len = strlen(path);
            if (len > 0 && path[len-1] == '\n') path[len-1] = '\0';
            
            if (!fgets(line, sizeof(line), stdin)) break;
            char *rows_csv = line;
            len = strlen(rows_csv);
            if (len > 0 && rows_csv[len-1] == '\n') rows_csv[len-1] = '\0';
            
            if (!fgets(line, sizeof(line), stdin)) break;
            block_rows_override = (uint32_t)atoi(line);
            if (block_rows_override == 0 || block_rows_override > MAX_BLOCK_ROWS) {
                fprintf(stderr, "Error: invalid block_rows\n");
                continue;
            }
            
            /* Process this command */
            if (process_file(path, rows_csv, block_rows_override, binary, 1) != 0) {
                return 1;
            }
            /* Flush stdout after each command to ensure output is available to reader */
            fflush(stdout);
        }
        return 0;
    }

    /* Normal mode: command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s [--binary] [--block-rows N] <archive.zdata> <rows_csv>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[argi], "--binary") == 0) {
        binary = 1;
        argi++;
    }

    if (argi < argc - 2 && strcmp(argv[argi], "--block-rows") == 0) {
        argi++;
        if (argi >= argc) {
            fprintf(stderr, "Error: --block-rows requires a value\n");
            return 1;
        }
        block_rows_override = (uint32_t)atoi(argv[argi++]);
        if (block_rows_override == 0 || block_rows_override > MAX_BLOCK_ROWS) {
            fprintf(stderr, "Error: block_rows must be between 1 and %d, got %u\n", MAX_BLOCK_ROWS, block_rows_override);
            return 1;
        }
    }

    const char *path = argv[argi++];
    const char *rows_csv = argv[argi++];
    
    return process_file(path, rows_csv, block_rows_override, binary, 0);
}

static int process_file(const char *path, const char *rows_csv, uint32_t block_rows_override, int binary, int is_stdin_mode) {
    uint32_t rows_req[4096];
    int nreq = parse_rows_csv(rows_csv, rows_req, 4096);
    if (nreq <= 0) { fprintf(stderr, "No rows parsed\n"); return 1; }

    for (int i = 0; i < nreq; i++) {
        if (rows_req[i] < 0 || rows_req[i] >= MAX_ROWS) {
            fprintf(stderr, "Row out of range: %d\n", rows_req[i]);
            return 1;
        }
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("open archive"); return 1; }
    
    /* Set input file to fully buffered for better performance (reduces system calls) */
    /* Use larger buffer (256KB) for better I/O performance, especially for column queries */
    setvbuf(fp, NULL, _IOFBF, 262144);  /* 256KB buffer */

    /* Set stdout to line buffered for persistent mode (ensures data is available immediately) */
    /* In stdin mode, we need immediate flushing so readers don't block */
    if (binary) {
        if (is_stdin_mode) {
            setvbuf(stdout, NULL, _IOLBF, 0);  /* Line buffered for stdin mode */
        } else {
            setvbuf(stdout, NULL, _IOFBF, 262144);  /* 256KB buffer for normal mode */
        }
    }

    ZSTD_seekable *zs = ZSTD_seekable_create();
    if (!zs) { fprintf(stderr, "ZSTD_seekable_create failed\n"); fclose(fp); return 1; }

    size_t ir = ZSTD_seekable_initFile(zs, fp);
    if (ZSTD_isError(ir)) {
        fprintf(stderr, "seekable_initFile error: %s\n", ZSTD_getErrorName(ir));
        ZSTD_seekable_free(zs);
        fclose(fp);
        return 1;
    }

    /* Determine ncols and block_rows by reading the first referenced block */
    uint32_t block_rows;
    unsigned first_block;
    uint8_t *frameBuf = NULL;
    size_t frameSz = 0;
    size_t frameBufCapacity = 0;  /* Track buffer capacity for reuse */

    if (block_rows_override > 0) {
        /* Use provided block_rows from metadata */
        block_rows = block_rows_override;
        first_block = (unsigned)(rows_req[0] / block_rows);
        if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCapacity)) {
            if (frameBuf) free(frameBuf);
            ZSTD_seekable_free(zs);
            fclose(fp);
            return 1;
        }
    } else {
        /* Auto-detect: try common values: 1, 4, 16 */
        /* For column queries (X_CM), block_rows is typically 1 (block_columns=1) */
        /* For row queries (X_RM), block_rows is typically 4 or 16 */
        block_rows = 1;  /* Start with 1 (most common for column queries) */
        first_block = (unsigned)(rows_req[0] / block_rows);
        if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCapacity)) {
            /* If block_rows=1 fails, try block_rows=4 (common for row queries) */
            if (frameBuf) free(frameBuf);
            frameBuf = NULL;
            frameBufCapacity = 0;
            block_rows = 4;
            first_block = (unsigned)(rows_req[0] / block_rows);
            if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCapacity)) {
                /* If block_rows=4 fails, try block_rows=16 (default) */
                if (frameBuf) free(frameBuf);
                frameBuf = NULL;
                frameBufCapacity = 0;
                block_rows = 16;
                first_block = (unsigned)(rows_req[0] / block_rows);
                if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCapacity)) {
                    if (frameBuf) free(frameBuf);
                    ZSTD_seekable_free(zs);
                    fclose(fp);
                    return 1;
                }
            }
        }
    }

    BlockHeader hdr;
    const uint32_t *indptr = NULL;
    const uint32_t *indices = NULL;
    const uint16_t *data = NULL;

    if (!parse_block(frameBuf, frameSz, &hdr, &indptr, &indices, &data, &block_rows)) {
        fprintf(stderr, "Failed to parse first block\n");
        free(frameBuf);
        ZSTD_seekable_free(zs);
        fclose(fp);
        return 1;
    }

    uint32_t ncols_out = hdr.ncols;

    if (binary) {
        write_le32(stdout, (uint32_t)nreq);
        write_le32(stdout, ncols_out);
    }

    /* Process rows in order (already ascending, so naturally groups by block) */
    /* Cache one block at a time - each block is only decompressed once */
    unsigned cached_block = first_block;
    
    /* Pre-sort rows by block to maximize block reuse (rows are already sorted, but this groups by block) */
    /* This is already done by the caller, so we can rely on rows being in ascending order */

    for (int i = 0; i < nreq; i++) {
        uint32_t row = rows_req[i];
        unsigned block_id = (unsigned)(row / block_rows);
        
        /* Decompress block only if we haven't seen it yet */
        if (block_id != cached_block) {
            if (!decompress_frame(zs, block_id, &frameBuf, &frameSz, &frameBufCapacity)) {
                if (frameBuf) free(frameBuf);
                ZSTD_seekable_free(zs);
                fclose(fp);
                return 1;
            }
            if (!parse_block(frameBuf, frameSz, &hdr, &indptr, &indices, &data, NULL)) {
                fprintf(stderr, "Failed to parse block %u\n", block_id);
                if (frameBuf) free(frameBuf);
                ZSTD_seekable_free(zs);
                fclose(fp);
                return 1;
            }
            cached_block = block_id;
        }

        /* Extract row data - optimize: compute r directly without block_start */
        uint32_t r = row - (block_id * block_rows);
        
        /* Validate row is within the block's actual row count */
        if (r >= hdr.nrows_in_block) {
            fprintf(stderr, "Row %u is beyond block %u's row count (%u)\n", row, block_id, hdr.nrows_in_block);
            if (frameBuf) free(frameBuf);
            ZSTD_seekable_free(zs);
            fclose(fp);
            return 1;
        }
        
        uint32_t p0 = indptr[r];
        uint32_t p1 = indptr[r + 1];
        uint32_t nnz = p1 - p0;

        /* Output immediately (rows are already in correct order) */
        if (!binary) {
            printf("row %u nnz %u:", row, nnz);
            for (uint32_t p = p0; p < p1; p++) {
                printf(" %u:%u", indices[p], data[p]);
            }
            printf("\n");
        } else {
            /* Write row header */
            write_le32(stdout, row);
            write_le32(stdout, nnz);
            /* Write indices and data arrays in bulk (optimized: combine when possible) */
            if (nnz > 0) {
                fwrite(indices + p0, sizeof(uint32_t), nnz, stdout);
                fwrite(data + p0, sizeof(uint16_t), nnz, stdout);
            }
        }
    }

    if (frameBuf) free(frameBuf);
    ZSTD_seekable_free(zs);
    fclose(fp);
    return 0;
}
