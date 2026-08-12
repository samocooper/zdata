#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "zstd.h"
#include "zstd_seekable.h"

#define MAX_BLOCK_ROWS 256
#define MAX_ROWS       1000000
#define INITIAL_ROWS   8192

/* -----------------------------------------------------------------------
   Abstract dtype system. DTypeRead and DTYPE_READ_TABLE are generated from
   zdata/dtypes.py -- the single source of truth shared with the Python layer
   and with mtx_to_zdata.c. Regenerate with:
       python -m zdata.dtypes --write-header
   ----------------------------------------------------------------------- */
#include "dtype_table.h"

#define DTYPE_TABLE DTYPE_READ_TABLE

/* IEEE-754 binary16 -> float. Mirrors float_to_half_bits() in mtx_to_zdata.c;
   native _Float16 where the compiler has it, explicit bit decode otherwise. */
#if defined(__FLT16_MAX__) && !defined(ZDATA_NO_NATIVE_FLOAT16)
#  define ZDATA_HAVE_NATIVE_FLOAT16 1
#endif

static float half_bits_to_float(uint16_t bits) {
#ifdef ZDATA_HAVE_NATIVE_FLOAT16
    _Float16 h;
    memcpy(&h, &bits, sizeof h);
    return (float)h;
#else
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t expo = (bits >> 10) & 0x1Fu;
    uint32_t mant = bits & 0x3FFu;
    uint32_t out;
    if (expo == 0u) {
        if (mant == 0u) { out = sign; }             /* +/-0 */
        else {                                       /* subnormal -> normalise */
            expo = 127 - 15 + 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; expo--; }
            mant &= 0x3FFu;
            out = sign | (expo << 23) | (mant << 13);
        }
    } else if (expo == 0x1Fu) {                      /* Inf / NaN */
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((expo - 15 + 127) << 23) | (mant << 13);
    }
    {
        float f;
        memcpy(&f, &out, sizeof f);
        return f;
    }
#endif
}
#define NUM_DTYPES ZDATA_NUM_DTYPES

static const DTypeRead *dtype_by_version(uint32_t ver) {
    for (size_t i = 0; i < NUM_DTYPES; i++)
        if (DTYPE_TABLE[i].version == ver) return &DTYPE_TABLE[i];
    return NULL;
}

/* -----------------------------------------------------------------------
   Helpers
   ----------------------------------------------------------------------- */
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

typedef struct {
    uint32_t magic, version, start_row, nrows_in_block, ncols, nnz;
} BlockHeader;

static int parse_rows_csv(const char *s, uint32_t **out_ptr, int *capacity) {
    int n = 0;
    const char *p = s;
    uint32_t *out = *out_ptr;
    int cap = *capacity;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p) break;
        if (v <= UINT32_MAX) {
            if (n >= cap) {
                cap *= 2;
                uint32_t *tmp = (uint32_t *)realloc(out, (size_t)cap * sizeof(uint32_t));
                if (!tmp) { fprintf(stderr, "OOM\n"); return -1; }
                out = tmp;
            }
            out[n++] = (uint32_t)v;
        }
        p = end;
    }
    *out_ptr = out;
    *capacity = cap;
    return n;
}

/* Parse a decompressed block.  Type-agnostic: val_size is derived from the
   version field in the block header. */
static int parse_block(const uint8_t *buf, size_t sz,
                       BlockHeader *hdr,
                       const uint32_t **indptr,
                       const uint32_t **indices,
                       const void **data,
                       uint32_t *block_rows_out) {
    if (sz < 24 + 4) return 0;

    hdr->magic           = read_le32(buf + 0);
    hdr->version         = read_le32(buf + 4);
    hdr->start_row       = read_le32(buf + 8);
    hdr->nrows_in_block  = read_le32(buf + 12);
    hdr->ncols           = read_le32(buf + 16);
    hdr->nnz             = read_le32(buf + 20);

    if (hdr->magic != 0x5253435A || hdr->nrows_in_block == 0 ||
        hdr->nrows_in_block > MAX_BLOCK_ROWS)
        return 0;

    const DTypeRead *dt = dtype_by_version(hdr->version);
    if (!dt) return 0;   /* unknown version */

    size_t val_size = dt->val_size;
    size_t remaining = sz - 24;
    size_t data_indices_bytes = (size_t)hdr->nnz * (sizeof(uint32_t) + val_size);
    if (remaining < data_indices_bytes) return 0;

    size_t indptr_bytes = remaining - data_indices_bytes;
    if (indptr_bytes % 4 != 0 || indptr_bytes < 4) return 0;

    uint32_t inferred_block_rows = (uint32_t)(indptr_bytes / 4) - 1;
    if (inferred_block_rows > MAX_BLOCK_ROWS || inferred_block_rows < hdr->nrows_in_block)
        return 0;

    if (block_rows_out) *block_rows_out = inferred_block_rows;

    size_t off = 24;
    *indptr = (const uint32_t*)(buf + off);
    off += (inferred_block_rows + 1) * sizeof(uint32_t);

    size_t idx_bytes  = (size_t)hdr->nnz * sizeof(uint32_t);
    size_t data_bytes = (size_t)hdr->nnz * val_size;
    if (off + idx_bytes + data_bytes > sz) return 0;

    *indices = (const uint32_t*)(buf + off);
    off += idx_bytes;
    *data = (const void*)(buf + off);
    return 1;
}

static int decompress_frame(ZSTD_seekable *zs, unsigned frameIndex,
                            uint8_t **outBuf, size_t *outSize, size_t *bufCapacity) {
    size_t dsz = ZSTD_seekable_getFrameDecompressedSize(zs, frameIndex);
    if (ZSTD_isError(dsz)) {
        fprintf(stderr, "getFrameDecompressedSize error: %s\n", ZSTD_getErrorName(dsz));
        return 0;
    }
    if (*outBuf == NULL || *bufCapacity < dsz) {
        if (*outBuf) free(*outBuf);
        *outBuf = (uint8_t*)malloc(dsz);
        if (!*outBuf) { fprintf(stderr, "OOM (%zu)\n", dsz); return 0; }
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

/* Print a single value from the data array in text mode. */
static void print_value(const void *data, uint32_t idx, const DTypeRead *dt) {
    switch (dt->version) {
    case 2:  printf("%u",    ((const uint16_t*)data)[idx]); break;
    case 3:  printf("%.6g",  ((const float*)data)[idx]);    break;
    case 4:  printf("%u",    ((const uint8_t*)data)[idx]);  break;
    case 5:  printf("%u",    ((const uint32_t*)data)[idx]); break;
    case 6:  printf("%llu",  (unsigned long long)((const uint64_t*)data)[idx]); break;
    case 7:  printf("%d",    ((const int8_t*)data)[idx]);   break;
    case 8:  printf("%d",    ((const int16_t*)data)[idx]);  break;
    case 9:  printf("%d",    ((const int32_t*)data)[idx]);  break;
    case 10: printf("%lld",  (long long)((const int64_t*)data)[idx]); break;
    case 11: printf("%.15g", ((const double*)data)[idx]);   break;
    case 12: printf("%.4g",  half_bits_to_float(((const uint16_t*)data)[idx])); break;
    default: printf("?"); break;
    }
}

/* Forward declaration */
static int process_file(const char *path, const char *rows_csv,
                        uint32_t block_rows_override, int binary, int is_stdin_mode);

int main(int argc, char *argv[]) {
    int binary = 0;
    int argi = 1;
    uint32_t block_rows_override = 0;
    int read_from_stdin = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--stdin") == 0) {
            read_from_stdin = 1;
            for (int j = 1; j < argc; j++)
                if (strcmp(argv[j], "--binary") == 0) binary = 1;
            break;
        }
    }

    if (read_from_stdin) {
        char path_buf[4096], rows_buf[65536], block_buf[64];
        while (fgets(path_buf, sizeof(path_buf), stdin)) {
            size_t len = strlen(path_buf);
            if (len > 0 && path_buf[len-1] == '\n') path_buf[len-1] = '\0';
            if (!fgets(rows_buf, sizeof(rows_buf), stdin)) break;
            len = strlen(rows_buf);
            if (len > 0 && rows_buf[len-1] == '\n') rows_buf[len-1] = '\0';
            if (!fgets(block_buf, sizeof(block_buf), stdin)) break;
            block_rows_override = (uint32_t)atoi(block_buf);
            if (block_rows_override == 0 || block_rows_override > MAX_BLOCK_ROWS) {
                fprintf(stderr, "Error: invalid block_rows\n"); continue;
            }
            if (process_file(path_buf, rows_buf, block_rows_override, binary, 1) != 0) return 1;
            fflush(stdout);
        }
        return 0;
    }

    if (argc < 3) {
        fprintf(stderr, "Usage: %s [--binary] [--block-rows N] <archive.zdata> <rows_csv>\n"
                        "       %s --stdin [--binary]\n", argv[0], argv[0]);
        return 1;
    }

    if (strcmp(argv[argi], "--binary") == 0) { binary = 1; argi++; }
    if (argi < argc - 2 && strcmp(argv[argi], "--block-rows") == 0) {
        argi++;
        block_rows_override = (uint32_t)atoi(argv[argi++]);
        if (block_rows_override == 0 || block_rows_override > MAX_BLOCK_ROWS) {
            fprintf(stderr, "block_rows must be 1–%d\n", MAX_BLOCK_ROWS); return 1;
        }
    }

    const char *path = argv[argi++];
    const char *rows_csv = argv[argi++];
    return process_file(path, rows_csv, block_rows_override, binary, 0);
}

static int process_file(const char *path, const char *rows_csv,
                        uint32_t block_rows_override, int binary, int is_stdin_mode) {
    int rows_cap = INITIAL_ROWS;
    uint32_t *rows_req = (uint32_t *)malloc((size_t)rows_cap * sizeof(uint32_t));
    if (!rows_req) { fprintf(stderr, "OOM\n"); return 1; }

    int nreq = parse_rows_csv(rows_csv, &rows_req, &rows_cap);
    if (nreq <= 0) { fprintf(stderr, "No rows parsed\n"); free(rows_req); return 1; }
    for (int i = 0; i < nreq; i++)
        if (rows_req[i] >= MAX_ROWS) { fprintf(stderr, "Row out of range: %u\n", rows_req[i]); free(rows_req); return 1; }

    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("open archive"); free(rows_req); return 1; }
    setvbuf(fp, NULL, _IOFBF, 262144);

    if (binary) {
        if (is_stdin_mode) setvbuf(stdout, NULL, _IOLBF, 0);
        else               setvbuf(stdout, NULL, _IOFBF, 262144);
    }

    ZSTD_seekable *zs = ZSTD_seekable_create();
    if (!zs) { fprintf(stderr, "seekable_create failed\n"); fclose(fp); free(rows_req); return 1; }
    size_t ir = ZSTD_seekable_initFile(zs, fp);
    if (ZSTD_isError(ir)) {
        fprintf(stderr, "initFile error: %s\n", ZSTD_getErrorName(ir));
        ZSTD_seekable_free(zs); fclose(fp); free(rows_req); return 1;
    }

    /* Determine block_rows by reading the first referenced block */
    uint32_t block_rows;
    unsigned first_block;
    uint8_t *frameBuf = NULL;
    size_t frameSz = 0, frameBufCap = 0;

    if (block_rows_override > 0) {
        block_rows = block_rows_override;
        first_block = (unsigned)(rows_req[0] / block_rows);
        if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCap))
            goto fail;
    } else {
        /* Auto-detect */
        block_rows = 1;
        first_block = (unsigned)(rows_req[0] / block_rows);
        if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCap)) {
            free(frameBuf); frameBuf = NULL; frameBufCap = 0;
            block_rows = 4;
            first_block = (unsigned)(rows_req[0] / block_rows);
            if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCap)) {
                free(frameBuf); frameBuf = NULL; frameBufCap = 0;
                block_rows = 16;
                first_block = (unsigned)(rows_req[0] / block_rows);
                if (!decompress_frame(zs, first_block, &frameBuf, &frameSz, &frameBufCap))
                    goto fail;
            }
        }
    }

    BlockHeader hdr;
    const uint32_t *indptr = NULL, *indices = NULL;
    const void *data = NULL;

    if (!parse_block(frameBuf, frameSz, &hdr, &indptr, &indices, &data, &block_rows)) {
        fprintf(stderr, "Failed to parse first block\n"); goto fail;
    }

    uint32_t ncols_out = hdr.ncols;
    uint32_t detected_version = hdr.version;
    const DTypeRead *dt = dtype_by_version(detected_version);
    if (!dt) { fprintf(stderr, "Unknown version %u\n", detected_version); goto fail; }
    size_t val_size = dt->val_size;

    if (binary) {
        write_le32(stdout, (uint32_t)nreq);
        write_le32(stdout, ncols_out);
        write_le32(stdout, detected_version);
    }

    unsigned cached_block = first_block;

    for (int i = 0; i < nreq; i++) {
        uint32_t row = rows_req[i];
        unsigned block_id = (unsigned)(row / block_rows);

        if (block_id != cached_block) {
            if (!decompress_frame(zs, block_id, &frameBuf, &frameSz, &frameBufCap)) goto fail;
            if (!parse_block(frameBuf, frameSz, &hdr, &indptr, &indices, &data, NULL)) {
                fprintf(stderr, "Failed to parse block %u\n", block_id); goto fail;
            }
            cached_block = block_id;
        }

        uint32_t r = row - (block_id * block_rows);
        if (r >= hdr.nrows_in_block) {
            fprintf(stderr, "Row %u beyond block %u's row count (%u)\n", row, block_id, hdr.nrows_in_block);
            goto fail;
        }

        uint32_t p0 = indptr[r], p1 = indptr[r + 1], nnz = p1 - p0;

        if (!binary) {
            printf("row %u nnz %u:", row, nnz);
            for (uint32_t p = p0; p < p1; p++) {
                printf(" %u:", indices[p]);
                print_value(data, p, dt);
            }
            printf("\n");
        } else {
            write_le32(stdout, row);
            write_le32(stdout, nnz);
            if (nnz > 0) {
                fwrite(indices + p0, sizeof(uint32_t), nnz, stdout);
                fwrite((const uint8_t*)data + p0 * val_size, val_size, nnz, stdout);
            }
        }
    }

    free(frameBuf);
    ZSTD_seekable_free(zs);
    fclose(fp);
    free(rows_req);
    return 0;

fail:
    if (frameBuf) free(frameBuf);
    ZSTD_seekable_free(zs);
    fclose(fp);
    free(rows_req);
    return 1;
}
