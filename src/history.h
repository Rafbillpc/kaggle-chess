#pragma once

#include "types.h"

constexpr int CORRHIST_SIZE = 1<<13;

constexpr int CORRHIST_LIMIT = 1024;

// [color][from to]
using MainHistory  = int16_t[COLOR_NB][SQUARE_NB * SQUARE_NB];

// [piece to][piece_type]
using CaptureHistory = int16_t[PIECE_NB * SQUARE_NB][PIECE_TYPE_NB];

// [piece to]
using CounterMoveHistory = Move[PIECE_NB * SQUARE_NB];

// [isCap][piece to][piece to]
constexpr int CH_bits = 15;
constexpr int CH_size = 1<<CH_bits;
constexpr int CH_mask = CH_size-1;

using ContinuationHistory = int16_t[2][CH_size];
struct ContinuationHistory_at {
  int16_t* table;
  uint32_t hash;

  inline
  int16_t& at(uint32_t mask2) {
    return table[(hash^mask2)&CH_mask];
  }

  inline
  int16_t const& at(uint32_t mask2) const {
    return table[(hash^mask2)&CH_mask];
  }
};

// [stm][pawn hash]
using PawnCorrHist = int16_t[2][CORRHIST_SIZE];

// [stm][pawn hash]
using NonPawnCorrHist = int16_t[2][CORRHIST_SIZE];

inline int getCorrHistIndex(Key pawnKey){
  return pawnKey % CORRHIST_SIZE;
}

inline void addToCorrhist(int16_t& history, int value){
  history += value - int(history) * myabs(value) / CORRHIST_LIMIT;
}

inline void addToHistory(int16_t& history, int value) {
  history += value - int(history) * myabs(value) / 16384;
}
