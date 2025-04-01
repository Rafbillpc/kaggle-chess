#include "nnue.h"
#include "bitboard.h"
#include "incbin.h"
#include "position.h"

// #include <cmath>

#ifdef SAVESAMPLES
#include <iostream>
#include <fstream>
#endif

#define AsVecI(x) *(VecI*)(&x)
#define AsVecF(x) *(VecF*)(&x)

namespace NNUE {

  constexpr int FloatInVec = sizeof(VecI) / sizeof(float);
  constexpr int I16InVec = sizeof(VecI) / sizeof(int16_t);
  constexpr int I8InVec = sizeof(VecI) / sizeof(int8_t);

  constexpr int FtShift = 9;
  
  alignas(Alignment) Net Weights;
  
#ifdef SAVESAMPLES
  std::ofstream SAMPLES;
#endif

  // For every possible uint16 number, store the count of active bits,
  // and the index of each active bit
  alignas(Alignment) uint16_t nnzTable[256][8];


  bool needRefresh(Color side, Square oldKing, Square newKing) {
    // Crossed half?
    if ((oldKing & 0b100) != (newKing & 0b100))
      return true;

    return false;
    // KingBucketsScheme[relative_square(side, oldKing)]
    //   != KingBucketsScheme[relative_square(side, newKing)];
  }

  inline VecI* featureAddress(Square kingSq, Color side, Piece pc, Square sq) {
    if (kingSq & 0b100)
      sq = Square(sq ^ 7);

    return (VecI*) Weights.FeatureWeights
      // [KingBucketsScheme[relative_square(side, kingSq)]]
      [side != piece_color(pc)]
      [piece_type(pc)-1]
      [relative_square(side, sq)];
  }

  template <int InputSize>
  inline void multiAdd(VecI* output, VecI* input, VecI* add0){
    for (int i = 0; i < InputSize / I16InVec; ++i) {
      output[i] = addEpi16(input[i], add0[i]);
    }
  }

  template <int InputSize>
  inline void multiSub(VecI* output, VecI* input, VecI* sub0){
    for (int i = 0; i < InputSize / I16InVec; ++i)
      output[i] = subEpi16(input[i], sub0[i]);
  }

  template <int InputSize>
  inline void multiAddAdd(VecI* output, VecI* input, VecI* add0, VecI* add1){
    for (int i = 0; i < InputSize / I16InVec; ++i)
      output[i] = addEpi16(input[i], addEpi16(add0[i], add1[i]));
  }

  template <int InputSize>
  inline void multiSubAdd(VecI* output, VecI* input, VecI* sub0, VecI* add0) {
    for (int i = 0; i < InputSize / I16InVec; ++i)
      output[i] = subEpi16(addEpi16(input[i], add0[i]), sub0[i]);
  }

  template <int InputSize>
  inline void multiSubAddSub(VecI* output, VecI* input, VecI* sub0, VecI* add0, VecI* sub1) {
    for (int i = 0; i < InputSize / I16InVec; ++i)
      output[i] = subEpi16(addEpi16(input[i], add0[i]), addEpi16(sub0[i], sub1[i]));
  }

  template <int InputSize>
  inline void multiSubAddSubAdd(VecI* output, VecI* input, VecI* sub0, VecI* add0, VecI* sub1, VecI* add1) {
    for (int i = 0; i < InputSize / I16InVec; ++i)
      output[i] = addEpi16(input[i], subEpi16(addEpi16(add0[i], add1[i]), addEpi16(sub0[i], sub1[i])));
  }

  void Accumulator::addPiece(Square kingSq, Color side, Piece pc, Square sq) {
    multiAdd<L1>((VecI*) colors[side], (VecI*) colors[side], featureAddress(kingSq, side, pc, sq));
  }

  void Accumulator::removePiece(Square kingSq, Color side, Piece pc, Square sq) {
    multiSub<L1>((VecI*) colors[side], (VecI*) colors[side], featureAddress(kingSq, side, pc, sq));
  }

  void Accumulator::doUpdates(Square kingSq, Color side, Accumulator& input) {
    DirtyPieces dp = this->dirtyPieces;
    if (dp.type == DirtyPieces::CASTLING) 
      {
        multiSubAddSubAdd<L1>((VecI*) colors[side], (VecI*) input.colors[side], 
                              featureAddress(kingSq, side, dp.sub0.pc, dp.sub0.sq),
                              featureAddress(kingSq, side, dp.add0.pc, dp.add0.sq),
                              featureAddress(kingSq, side, dp.sub1.pc, dp.sub1.sq),
                              featureAddress(kingSq, side, dp.add1.pc, dp.add1.sq));
      } else if (dp.type == DirtyPieces::CAPTURE) 
      { 
        multiSubAddSub<L1>((VecI*) colors[side], (VecI*) input.colors[side], 
                           featureAddress(kingSq, side, dp.sub0.pc, dp.sub0.sq),
                           featureAddress(kingSq, side, dp.add0.pc, dp.add0.sq),
                           featureAddress(kingSq, side, dp.sub1.pc, dp.sub1.sq));
      } else
      {
        multiSubAdd<L1>((VecI*) colors[side], (VecI*) input.colors[side], 
                        featureAddress(kingSq, side, dp.sub0.pc, dp.sub0.sq),
                        featureAddress(kingSq, side, dp.add0.pc, dp.add0.sq));
      }
    updated[side] = true;
  }

  void Accumulator::reset(Color side) {
    memcpy(colors[side], Weights.FeatureBiases, sizeof(Weights.FeatureBiases));
  }

  void Accumulator::refresh(Position& pos, Color side) {
    reset(side);
    const Square kingSq = pos.kingSquare(side);
    Bitboard occupied = pos.pieces();
    while (occupied) {
      const Square sq = popLsb(occupied);
      addPiece(kingSq, side, pos.board[sq], sq);
    }
    updated[side] = true;
  }

  void FinnyEntry::reset() {
    memset(byColorBB, 0, sizeof(byColorBB));
    memset(byPieceBB, 0, sizeof(byPieceBB));
    acc.reset(WHITE);
    acc.reset(BLACK);
  }

  Score evaluate(Position& pos, Accumulator& accumulator) {
    constexpr int divisor = (32 + OutputBuckets - 1) / OutputBuckets;
    int bucket = (BitCount(pos.pieces()) - 2) / divisor;

#ifdef SAVESAMPLES 
    {
      int16_t nfeatures = 0;
      int16_t features[64];
      for(int side = 0; side < 2; ++ side) {
        const Square kingSq = pos.kingSquare((Color)side);
        Bitboard occupied = pos.pieces();
        while (occupied) {
          Square sq = popLsb(occupied);
          Piece pc = pos.board[sq];
          
          if (kingSq & 0b100)
            sq = Square(sq ^ 7);

          int f = (side^pos.sideToMove)*768
            + (side != piece_color(pc)) * 6*64
            + (piece_type(pc)-1) * 64
            + (relative_square((Color)side, sq));

          features[nfeatures++] = f;
        }
      }
      SAMPLES.write((char*)&nfeatures, sizeof(int16_t));
      SAMPLES.write((char*)features, nfeatures * sizeof(int16_t));
      SAMPLES.write((char*)&bucket, sizeof(int));
    }
#endif

    __m128i base = _mm_setzero_si128();
    __m128i lookupInc = _mm_set1_epi16(8);

    VecF vecfZero = setzeroPs();
    VecF vecfOne = set1Ps(1.0f);
    VecI veciZero = setzeroSi();
    VecI veciOne = set1Epi16(NetworkQA);

    // L1 propagation is int8 -> float, so we multiply 4 ft outputs at a time
    uint16_t nnzIndexes[L1 / 2];
    int nnzCount = 0;

    alignas(Alignment) uint8_t ftOut[L1*2];
    alignas(Alignment) float l1Out[L2];
    alignas(Alignment) float l2Out[L3];
    float l3Out;

    constexpr float L1Mul = 1.0f / float(NetworkQA * NetworkQA * NetworkQB >> FtShift);
    VecF L1MulVec = set1Ps(L1Mul);

    // for (int them = 0; them <= 1; ++them) 
    // {
    //   int16_t* acc = accumulator.colors[pos.sideToMove ^ them];
    //   for (int i = 0; i < L1; i += 1) 
    //   {
    //     std::cout << (float)acc[i] / NetworkQA << ' ';
    //   }
    //   std::cout << std::endl;
    // }
    
    // activate FT
    for (int them = 0; them <= 1; ++them) 
    {
      int16_t* acc = accumulator.colors[pos.sideToMove ^ them];
      for (int i = 0; i < L1; i += I8InVec) 
      {
        VecI c0 = minEpi16(maxEpi16(AsVecI(acc[i]), veciZero), veciOne);
        // VecI c1 = c0; // minEpi16(AsVecI(acc[i + L1/2]), veciOne);

        VecI d0 = minEpi16(maxEpi16(AsVecI(acc[i + I16InVec]), veciZero), veciOne);
        // VecI d1 = d0; // minEpi16(AsVecI(acc[i + L1/2 + I16InVec]), veciOne);

        VecI cProd = mulhiEpi16(slliEpi16(c0, 16 - FtShift), c0);
        VecI dProd = mulhiEpi16(slliEpi16(d0, 16 - FtShift), d0);

        VecI packed = packusEpi16(cProd, dProd);
        AsVecI(ftOut[them * L1 + i]) = packed;

        // a bit mask where each bit (x) is 1, if the xth int32 in the product is > 0
        uint16_t nnzMask = getNnzMask(packed);

        // Usually (in AVX2) only one lookup is needed, as there are 8 ints in a vec.
        for (int lookup = 0; lookup < FloatInVec; lookup += 8) {
          uint8_t slice = (nnzMask >> lookup) & 0xFF;
          __m128i indexes = _mm_loadu_si128((__m128i*)nnzTable[slice]);
          _mm_storeu_si128((__m128i*)(nnzIndexes + nnzCount), _mm_add_epi16(base, indexes));
          nnzCount += BitCount(slice);
          base = _mm_add_epi16(base, lookupInc);
        }
      }
    }

    // for (int i = 0; i < L1; i += 1) 
    //   {
    //     std::cout << (float)ftOut[i] / 128 << ' ';
    //   }
    // std::cout << std::endl;
    
    { // propagate l1

      alignas(Alignment) int32_t sums[L2];
      memset(sums, 0, sizeof(sums));

      for (int i = 0; i < nnzCount; i++) {
        int l1in = nnzIndexes[i]*4;
        VecI vecFtOut = set1Epi32( *(uint32_t*)(ftOut + l1in) );
        for (int j = 0; j < L2; j += FloatInVec) {
          VecI vecWeight = AsVecI(Weights.L1Weights[l1in + j/4]);
          AsVecI(sums[j]) = dpbusdEpi32(AsVecI(sums[j]), vecFtOut, vecWeight);
        }
      }

      // for (int i = 0; i < L2; i += 1) 
      //   {
      //     std::cout << (float)sums[i] << ' ';
      //   }
      // std::cout << std::endl;
      
      for (int i = 0; i < L2; i += FloatInVec) {
        VecF vecBias = AsVecF(Weights.L1Biases[i]);
        VecF casted = mulAddPs(castEpi32ToPs(AsVecI(sums[i])), L1MulVec, vecBias);
        VecF clipped = minPs(maxPs(casted, vecfZero), vecfOne);
        AsVecF(l1Out[i]) = mulPs(clipped, clipped);
      }
    }

    // for (int i = 0; i < L2; i += 1) 
    //   {
    //     std::cout << (float)l1Out[i] << ' ';
    //   }
    // std::cout << std::endl;

    { // propagate l2
      alignas(Alignment) float sums[L3];
      memcpy(sums, Weights.L2Biases, sizeof(sums));

      for (int i = 0; i < L2; ++i) {
        VecF vecL1Out = set1Ps(l1Out[i]);
        for (int j = 0; j < L3; j += FloatInVec)
          AsVecF(sums[j]) = mulAddPs(AsVecF(Weights.L2Weights[i][j]), vecL1Out, AsVecF(sums[j]));
      }

      for (int i = 0; i < L3; i += FloatInVec)
        AsVecF(l2Out[i]) = minPs(maxPs(AsVecF(sums[i]), vecfZero), vecfOne);
    }

    // for (int i = 0; i < L3; i += 1) 
    //   {
    //     std::cout << (float)l2Out[i] << ' ';
    //   }
    // std::cout << std::endl;
    
    { // propagate l3
      VecF sums = setzeroPs();
      for (int i = 0; i < L3; i += FloatInVec)
        sums = mulAddPs(AsVecF(l2Out[i]), AsVecF( Weights.L3Weights[bucket][i]), sums);

      l3Out = reduceAddPs(sums) + Weights.L3Biases[bucket];
    }

    // for(int b = 0; b < 8; ++ b) {
    //   float out = Weights.L3Biases[b];
    //   for(int i = 0; i < L3; ++i) out += Weights.L3Weights[b][i] * l2Out[i];
    //   std::cerr << b << " " << out << std::endl;
    // }

    
    // std::cout << bucket << std::endl;
    
    // std::cout << (float)l3Out;
    // std::cout << std::endl;
    
    return l3Out * NetworkScale;
  }

}
