/* GENERATED FILE -- DO NOT EDIT BY HAND.
 *
 * Regenerate with:  python -m zdata.dtypes --write-header
 * Source of truth:  zdata/dtypes.py
 *
 * Editing this file directly will be caught by
 * tests/test_core/test_dtype_table_sync.py.
 */
#ifndef ZDATA_DTYPE_TABLE_H
#define ZDATA_DTYPE_TABLE_H

#include <stddef.h>
#include <stdint.h>

#define ZDATA_DEFAULT_DTYPE_VERSION 2

/* Writer-side table (mtx_to_zdata.c): version <-> element size <-> name. */
typedef struct {
    uint32_t    version;
    size_t      val_size;
    const char *name;
} DTypeInfo;

static const DTypeInfo DTYPE_TABLE[] = {
    {  2, 2, "uint16" },   /* backward-compatible default */
    {  3, 4, "float32" },   /* backward-compatible */
    {  4, 1, "uint8" },
    {  5, 4, "uint32" },
    {  6, 8, "uint64" },
    {  7, 1, "int8" },
    {  8, 2, "int16" },
    {  9, 4, "int32" },
    { 10, 8, "int64" },
    { 11, 8, "float64" },
    { 12, 2, "float16" },   /* IEEE-754 binary16 */
};

/* Reader-side table (zdata_read.c): adds a printf format and a float flag. */
typedef struct {
    uint32_t    version;
    size_t      val_size;
    const char *name;
    const char *fmt;
    int         is_float;
} DTypeRead;

static const DTypeRead DTYPE_READ_TABLE[] = {
    {  2, 2, "uint16", "%u", 0 },
    {  3, 4, "float32", "%.6g", 1 },
    {  4, 1, "uint8", "%u", 0 },
    {  5, 4, "uint32", "%u", 0 },
    {  6, 8, "uint64", "%llu", 0 },
    {  7, 1, "int8", "%d", 0 },
    {  8, 2, "int16", "%d", 0 },
    {  9, 4, "int32", "%d", 0 },
    { 10, 8, "int64", "%lld", 0 },
    { 11, 8, "float64", "%.15g", 1 },
    { 12, 2, "float16", "%.4g", 1 },
};

#define ZDATA_NUM_DTYPES (sizeof(DTYPE_TABLE) / sizeof(DTYPE_TABLE[0]))

#endif /* ZDATA_DTYPE_TABLE_H */
