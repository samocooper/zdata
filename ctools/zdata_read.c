#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "zstd.h"
#include "zstd_seekable.h"

#define BLOCK_ROWS 16
#define MAX_ROWS   4096

static uint32_t read_le32(const uint8_t *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void write_le32(FILE *out, uint32_t v) {
    uint8_t b[4];
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
    fwrite(b, 1, 4, out);
}

static void write_lef32(FILE *out, float f) {
    uint32_t v;
    memcpy(&v, &f, 4);
    write_le32(out, v);
}

typedef struct {
    uint32_t magic, version, start_row, nrows_in_block, ncols, nnz;
} BlockHeader;

static int parse_rows_csv(const char *s, int32_t *out, int max_out) {
    int n = 0;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (n < max_out) out[n++] = (int32_t)v;
        p = end;
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p == ',') p++;
    }
    return n;
}

static int parse_block(const uint8_t *buf, size_t sz,
                       BlockHeader *hdr,
                       const int32_t **indptr,
                       const int32_t **indices,
                       const float **data) {
    if (sz < 24 + (BLOCK_ROWS + 1) * 4) return 0;

    hdr->magic = read_le32(buf + 0);
    hdr->version = read_le32(buf + 4);
    hdr->start_row = read_le32(buf + 8);
    hdr->nrows_in_block = read_le32(buf + 12);
    hdr->ncols = read_le32(buf + 16);
    hdr->nnz = read_le32(buf + 20);

    if (hdr->magic != 0x5253435A || hdr->version != 1 || hdr->nrows_in_block != BLOCK_ROWS) {
        return 0;
    }

    size_t off = 24;
    *indptr = (const int32_t*)(buf + off);
    off += (BLOCK_ROWS + 1) * sizeof(int32_t);

    size_t idx_bytes = (size_t)hdr->nnz * sizeof(int32_t);
    size_t data_bytes = (size_t)hdr->nnz * sizeof(float);
    if (off + idx_bytes + data_bytes > sz) return 0;

    *indices = (const int32_t*)(buf + off);
    off += idx_bytes;
    *data = (const float*)(buf + off);
    return 1;
}

static int decompress_frame(ZSTD_seekable *zs, unsigned frameIndex,
                            uint8_t **outBuf, size_t *outSize) {
    size_t dsz = ZSTD_seekable_getFrameDecompressedSize(zs, frameIndex);
    if (ZSTD_isError(dsz)) {
        fprintf(stderr, "getFrameDecompressedSize error: %s\n", ZSTD_getErrorName(dsz));
        return 0;
    }

    uint8_t *buf = (uint8_t*)malloc(dsz);
    if (!buf) {
        fprintf(stderr, "OOM frame buffer (%zu)\n", dsz);
        return 0;
    }

    size_t dr = ZSTD_seekable_decompressFrame(zs, buf, dsz, frameIndex);
    if (ZSTD_isError(dr)) {
        fprintf(stderr, "decompressFrame error: %s\n", ZSTD_getErrorName(dr));
        free(buf);
        return 0;
    }

    *outBuf = buf;
    *outSize = dr;
    return 1;
}

int main(int argc, char *argv[]) {
    int binary = 0;
    int argi = 1;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s [--binary] <archive.zdata> <rows_csv>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[argi], "--binary") == 0) {
        binary = 1;
        argi++;
    }

    const char *path = argv[argi++];
    const char *rows_csv = argv[argi++];

    int32_t rows_req[4096];
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

    ZSTD_seekable *zs = ZSTD_seekable_create();
    if (!zs) { fprintf(stderr, "ZSTD_seekable_create failed\n"); fclose(fp); return 1; }

    size_t ir = ZSTD_seekable_initFile(zs, fp);
    if (ZSTD_isError(ir)) {
        fprintf(stderr, "seekable_initFile error: %s\n", ZSTD_getErrorName(ir));
        ZSTD_seekable_free(zs);
        fclose(fp);
        return 1;
    }

    /* Determine ncols by reading the first referenced block */
    unsigned first_block = (unsigned)(rows_req[0] / BLOCK_ROWS);
    uint8_t *frameBuf = NULL;
    size_t frameSz = 0;

    if (!decompress_frame(zs, first_block, &frameBuf, &frameSz)) {
        ZSTD_seekable_free(zs);
        fclose(fp);
        return 1;
    }

    BlockHeader hdr;
    const int32_t *indptr = NULL;
    const int32_t *indices = NULL;
    const float *data = NULL;

    if (!parse_block(frameBuf, frameSz, &hdr, &indptr, &indices, &data)) {
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

    /* Cache one block at a time */
    unsigned cached_block = first_block;

    for (int i = 0; i < nreq; i++) {
        int32_t row = rows_req[i];
        unsigned block_id = (unsigned)(row / BLOCK_ROWS);
        int32_t block_start = (int32_t)block_id * BLOCK_ROWS;

        if (block_id != cached_block) {
            free(frameBuf);
            frameBuf = NULL;

            if (!decompress_frame(zs, block_id, &frameBuf, &frameSz)) {
                ZSTD_seekable_free(zs);
                fclose(fp);
                return 1;
            }
            if (!parse_block(frameBuf, frameSz, &hdr, &indptr, &indices, &data)) {
                fprintf(stderr, "Failed to parse block %u\n", block_id);
                free(frameBuf);
                ZSTD_seekable_free(zs);
                fclose(fp);
                return 1;
            }
            cached_block = block_id;
        }

        int32_t r = row - block_start;
        int32_t p0 = indptr[r];
        int32_t p1 = indptr[r + 1];
        int32_t nnz = p1 - p0;

        if (!binary) {
            printf("row %d nnz %d:", row, nnz);
            for (int32_t p = p0; p < p1; p++) printf(" %d:%.6g", indices[p], data[p]);
            printf("\n");
        } else {
            write_le32(stdout, (uint32_t)row);
            write_le32(stdout, (uint32_t)nnz);
            for (int32_t p = p0; p < p1; p++) write_le32(stdout, (uint32_t)indices[p]);
            for (int32_t p = p0; p < p1; p++) write_lef32(stdout, (float)data[p]);
        }
    }

    free(frameBuf);
    ZSTD_seekable_free(zs);
    fclose(fp);
    return 0;
}
