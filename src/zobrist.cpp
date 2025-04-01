#include "zobrist.h"

uint64_t ZOBRIST_TEMPO;
uint64_t ZOBRIST_PSQ[PIECE_NB][SQUARE_NB];
uint64_t ZOBRIST_EP[FILE_NB];
uint64_t ZOBRIST_CASTLING[16];
uint64_t ZOBRIST_50MR[120];

uint32_t CH_PIECE1[16];
uint32_t CH_SQUARE1[64];
uint32_t CH_PIECE2[16];
uint32_t CH_SQUARE2[64];

namespace Zobrist {

  struct xorshift64_state {
    uint64_t a;
  };

  [[clang::minsize]]
  __attribute__((noinline)) 
  uint64_t xorshift64(struct xorshift64_state *state)
  {
    uint64_t x = state->a;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state->a = x;
  }

  [[clang::minsize]]
  __attribute__((noinline)) 
  void init() {
    xorshift64_state gen;
    gen.a = 0x0F0F0F0F0F0F0F0Full;
    // std::mt19937_64 gen(12345);
    // std::uniform_int_distribution<uint64_t> dis;

    ZOBRIST_TEMPO = xorshift64(&gen);

    for (int pc = W_PAWN; pc < PIECE_NB; ++pc)
      for (Square sq = SQ_A1; sq < SQUARE_NB; ++sq)
        ZOBRIST_PSQ[pc][sq] = xorshift64(&gen);

    for (File f = FILE_A; f < FILE_NB; ++f)
      ZOBRIST_EP[f] = xorshift64(&gen);

    ZOBRIST_CASTLING[0] = 0;
    ZOBRIST_CASTLING[WHITE_OO] = xorshift64(&gen);
    ZOBRIST_CASTLING[WHITE_OOO] = xorshift64(&gen);
    ZOBRIST_CASTLING[BLACK_OO] = xorshift64(&gen);
    ZOBRIST_CASTLING[BLACK_OOO] = xorshift64(&gen);

    for (int i = 1; i < 16; i++) {

      if (BitCount(i) < 2)
        continue;

      uint64_t delta = 0;

      if (i & WHITE_OO)  delta ^= ZOBRIST_CASTLING[WHITE_OO];
      if (i & WHITE_OOO) delta ^= ZOBRIST_CASTLING[WHITE_OOO];
      if (i & BLACK_OO)  delta ^= ZOBRIST_CASTLING[BLACK_OO];
      if (i & BLACK_OOO) delta ^= ZOBRIST_CASTLING[BLACK_OOO];

      ZOBRIST_CASTLING[i] = delta;
    }

    // and now the 50mr stuff
    memset(ZOBRIST_50MR, 0, sizeof(ZOBRIST_50MR));
    for (int i = 14; i <= 100; i += 8) {
      uint64_t key = xorshift64(&gen);
      for (int j = 0; j < 8; j++)
        ZOBRIST_50MR[i+j] = key;
    }

    for(int i = 0; i < 16; ++i) CH_PIECE1[i] = xorshift64(&gen);
    for(int i = 0; i < 64; ++i) CH_SQUARE1[i] = xorshift64(&gen);
    for(int i = 0; i < 16; ++i) CH_PIECE2[i] = xorshift64(&gen);
    for(int i = 0; i < 64; ++i) CH_SQUARE2[i] = xorshift64(&gen);
  }

}
