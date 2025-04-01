#pragma once

#include "simd.h"
#include "types.h"

using namespace SIMD;

struct Position;

struct SquarePiece {
  Square sq;
  Piece pc;
};

struct DirtyPieces {
  SquarePiece sub0, add0, sub1, add1;

  enum {
    NORMAL, CAPTURE, CASTLING
  } type;
};

namespace NNUE {
  constexpr int FeaturesWidth = 768;
  constexpr int HL = 128;
  constexpr int L1 = HL;
  constexpr int L2 = 16;
  constexpr int L3 = 32;

  constexpr int OutputBuckets = 8;

  constexpr int NetworkScale = 400;
  constexpr int NetworkQA = 255;
  constexpr int NetworkQB = 128;

  struct PackedNet {
    float FeatureWeights[32][2][6][HL];
    float FeatureBiases[L1];

    float L1Weights[L1*2][L2];
    float L1Biases[L2];

    float L2Weights[L2][L3];
    float L2Biases[L3];

    float L3Weights[L3][OutputBuckets];
    float L3Biases[OutputBuckets];
  };

  struct Net {
    alignas(Alignment) int16_t FeatureWeights[2][6][64][L1];
    alignas(Alignment) int16_t FeatureBiases[L1];

    alignas(Alignment) int8_t L1Weights[L1*2][L2];
    alignas(Alignment) float L1Biases[L2];

    alignas(Alignment) float L2Weights[L2][L3];
    alignas(Alignment) float L2Biases[L3];

    alignas(Alignment) float L3Weights[OutputBuckets][L3];
    alignas(Alignment) float L3Biases[OutputBuckets];
  };

  extern Net Weights;
  extern uint16_t nnzTable[256][8];
  
  struct Accumulator {
    
    alignas(Alignment) int16_t colors[COLOR_NB][L1];

    bool updated[COLOR_NB];
    Square kings[COLOR_NB];
    DirtyPieces dirtyPieces;
    
    void addPiece(Square kingSq, Color side, Piece pc, Square sq);
    
    void removePiece(Square kingSq, Color side, Piece pc, Square sq);
    
    void doUpdates(Square kingSq, Color side, Accumulator& input);

    void reset(Color side);

    void refresh(Position& pos, Color side);
  };

  struct NNZEntry {
    uint16_t indexes[8];
  };

  struct FinnyEntry {
    Bitboard byColorBB[COLOR_NB][COLOR_NB];
    Bitboard byPieceBB[COLOR_NB][PIECE_TYPE_NB];
    Accumulator acc;

    void reset();
  };

  using FinnyTable = FinnyEntry[2];

  bool needRefresh(Color side, Square oldKing, Square newKing);

  void loadWeights();

  Score evaluate(Position& pos, Accumulator& accumulator);
}
