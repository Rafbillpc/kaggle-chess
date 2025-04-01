#pragma once

#include "types.h"

extern uint64_t ZOBRIST_TEMPO;
extern uint64_t ZOBRIST_PSQ[PIECE_NB][SQUARE_NB];
extern uint64_t ZOBRIST_EP[FILE_NB];
extern uint64_t ZOBRIST_CASTLING[16];
extern uint64_t ZOBRIST_50MR[120];

extern uint32_t CH_PIECE1[16];
extern uint32_t CH_SQUARE1[64];
extern uint32_t CH_PIECE2[16];
extern uint32_t CH_SQUARE2[64];

namespace Zobrist {

  void init();

}
