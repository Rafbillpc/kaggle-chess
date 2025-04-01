#pragma once
#include "header.hpp"

/*
 * This file contains the implementation of a simple range encoder/decoder (also
 * known as arithmetic coder, see https://en.wikipedia.org/wiki/Range_coding,
 * https://en.wikipedia.org/wiki/Arithmetic_coding).
 *
 * The code is based on https://github.com/thecbloom/recip_arith/.
 *
 */

/*
 * Example usage:
 *
 *  uint8_t buffer[1'000'000];
 *
 *  range_encoder out(buffer);
 *
 *  out.put(3,4,10); // write a symbol that corresponds to the subrange [3..6] of [0..9].
 *  out.put(5,1,6); // write a symbol that corresponds to the subrange [5..5] or [0..5].
 *  out.finish();
 *
 *  range_decoder in(buffer);
 *  int a = in.get(10);       // Get the next value a in the range [0..9].
 *  assert(3 <= a && a <= 6); // a lies in the range [3..6].
 *  in.update(3,4,10);        // notify the decoder that the last symbol did correspond to the subrange [3..6].
 *  int b = in.get(6);        // Get the next value b in the range [5..5].
 *  assert(a == 5);           // b lies in the range [5..5].
 *  in.update(5,1,6);         // notify the decoder that the last symbol did correspond to the subrange [5..5].
 *
 */

struct range_encoder {
  uint32_t low,range;
  uint8_t* ptr;
  uint32_t size;

  range_encoder() = default;

  FORCE_INLINE
  range_encoder(uint8_t* ptr_) {
    low   = 0;
    range = ~(uint32_t)0;
    ptr   = ptr_;
    size  = 0;
  }

  FORCE_INLINE
  void renorm(){
    while(range < (1<<24)) {
      *ptr++ = (uint8_t)(low>>24);
      size += 1;
      low   <<= 8;
      range <<= 8;
    }
  }

  FORCE_INLINE
  void carry() {
    uint8_t* p = ptr;
    do {
      --p;
      *p += 1;
    } while(*p == 0);
  }

  FORCE_INLINE
  void finish() {
    if(range > (1<<25)) {
      uint32_t code = low + (1<<24);
      if(code < low) carry();
      *ptr++ = (uint8_t)(code>>24);
      size += 1;
    }else{
      uint32_t code = low + (1<<16);
      if(code < low) carry();
      *ptr++ = (uint8_t)(code>>24);
      *ptr++ = (uint8_t)(code>>16);
      size += 2;
    }
  }

  FORCE_INLINE
  void put(uint32_t cdf_low, uint32_t cdf_size, uint32_t cdf_total) {
    renorm();
    uint32_t base_range = range/cdf_total;
    uint32_t save_low   = low;
    low += cdf_low * base_range;
    range = cdf_size * base_range;
    if(low < save_low) carry();
  }
};

struct range_decoder {
  uint32_t code,range;
  uint8_t const* ptr;

  FORCE_INLINE
  range_decoder(uint8_t const* ptr_) {
    ptr = ptr_;
    range = ~(uint32_t)0;
    code = *ptr++;
    code <<= 8; code |= *ptr++;
    code <<= 8; code |= *ptr++;
    code <<= 8; code |= *ptr++;
  }

  FORCE_INLINE
  void renorm() {
    while(range < (1<<24)) {
      code <<= 8;
      code |= *ptr++;
      range <<= 8;
    }
  }

  FORCE_INLINE
  uint32_t get(uint32_t cdf_total) {
    renorm();
    range = range/cdf_total;
    return code / range;
  }

  FORCE_INLINE
  void update(uint32_t cdf_low, uint32_t cdf_size, uint32_t cdf_total) {
    code -= range * cdf_low;
    range = range * cdf_size;
  }
};

