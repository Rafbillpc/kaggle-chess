#include "nnue.h"
#include "bitboard.h"
#include "incbin.h"
#include "position.h"
#include "util.h"

#include "range_coding.hpp"

#ifndef SHARED
#include <iostream>
#include <fstream>
#endif

INCBIN(EmbeddedNNUE, EvalFile);

namespace NNUE {

#ifdef SAVESAMPLES
  extern std::ofstream SAMPLES;
#endif
  
  int myround(float x) {
    int a;
    if(x > 0) {
      a = (int)(x + 0.5f);
    }else{
      a = -(int)(-x + 0.5f);
    }
    return a;
  }
  
  int16_t quant_a(float x) {
    float scaled = x * (float)NetworkQA;
    return (int)(myround(scaled));
  }
  
  int8_t quant_b(float x) {
    float scaled = x * (float)NetworkQB;
    return std::clamp<int>(myround(scaled), -NetworkQB+1, NetworkQB-1);
  }

  int unmap_square(int sq) {
    int rb = sq / 32;
    int b = sq / 8 % 4;
    int a = sq % 8;
    if(rb) { b = 3-b; b += 4; }
    return a*8+b;
  }

  int shuffle_2(int ix, int m) {
    int a = ix / (HL/2);
    ix %= (HL/2);
    a ^= m;
    return a*(HL/2) + ix;
  }
  
  void from_packed(Net& net, PackedNet const& pnet) {
    memset(&net, 0, sizeof(Net));
    
    for(int c = 0; c < 32; ++c) 
      for(int a = 0; a < 2; ++a) 
        for(int b = 0; b < 6; ++b) 
          for(int o = 0; o < HL; ++o)
            for(int m = 0; m < 2; ++m)
              {
                int sq = unmap_square(c+32*m);
                int x = sq/8;
                int y = sq%8;

                // std::cout << (c / (2*WINDOW_SIZE)) << ' ' << (c % (2*WINDOW_SIZE)) << std::endl;
                // std::cout << c <<"  " << m << " " << x << " " << y << std::endl;

                if(0 <= y && y < 8) {
                  net.FeatureWeights[a][b][8*x+y][shuffle_2(o,m)]
                    = quant_a(pnet.FeatureWeights[c][a][b][o]);
                }
              }

    for(int o = 0; o < L1; ++o) 
      net.FeatureBiases[o] = quant_a(pnet.FeatureBiases[o]);

    for(int o = 0; o < L2; ++o) {
      for(int i = 0; i < 2*L1; ++i) {
        net.L1Weights[(i/4)*4 + (o/4)][(o%4)*4 + (i%4)] = quant_b(pnet.L1Weights[i][o]);
      }
      net.L1Biases[o] = pnet.L1Biases[o];
    }

    for(int o = 0; o < L3; ++o) {
      for(int i = 0; i < L2; ++i) {
        net.L2Weights[i][o] = pnet.L2Weights[i][o];
      }
      net.L2Biases[o] = pnet.L2Biases[o];
    }
    for(int b = 0; b < OutputBuckets; ++b) {
      for(int i = 0; i < L3; ++i) {
        net.L3Weights[b][i] = pnet.L3Weights[i][b];
      }
      net.L3Biases[b] = pnet.L3Biases[b];
    }
  }

#define FOR(i, n) for(int i = 0; i < (n); ++i)
  
#ifdef USE_QUANT
  void load_packed_quant(PackedNet& pnet) {
    constexpr int NA = NetworkQA*2;
    constexpr int NB = NetworkQB;

    range_decoder dec(gEmbeddedNNUEData);

    int trit_a[2*NA+1];
    FOR(i, 2*NA+1) {
      trit_a[i] = dec.get(3);
      dec.update(trit_a[i], 1, 3);
    }

    int root_a[2*NA+1];
    FOR(i, 2*NA+1) {
      root_a[i] = i;
      if(trit_a[i] == 0) {
        while(trit_a[root_a[i]] != 1) root_a[i] += 1;
      }else if(trit_a[i] == 2) {
        while(trit_a[root_a[i]] != 1) root_a[i] -= 1;
      }
    }
    
    int count_a[2*NA+1];
    memset(count_a, 0, sizeof(count_a));
    
    int max_a = dec.get(1<<16);
    dec.update(max_a, 1, (1<<16));
    FOR(i, 2*NA+1) if(i == root_a[i]) {
      count_a[i] = dec.get(max_a+1);
      dec.update(count_a[i], 1, max_a+1);
      count_a[i] = count_a[i]*count_a[i];
    }

    int cdf_a[2*NA+2];
    cdf_a[0] = 0;
    FOR(i, 2*NA+1) cdf_a[i+1] = cdf_a[i] + count_a[i];
    auto decode_a = [&]() -> int {
      int v = dec.get(cdf_a[2*NA+1]);
      int x = 0;
      while(v >= cdf_a[x+1]) x += 1;
      dec.update(cdf_a[x], count_a[x], cdf_a[2*NA+1]);
      return x - NA;
    };
    
    FOR(a, 32) FOR(b, 2) FOR(c, 6) FOR(d, HL) {
      pnet.FeatureWeights[a][b][c][d] = 1.0 * decode_a() / NetworkQA;
    }
    FOR(o, L1) {
      pnet.FeatureBiases[o] = 1.0 * decode_a() / NetworkQA;
    }

    int trit_b[2*NB+1];
    FOR(i, 2*NB+1) {
      trit_b[i] = dec.get(3);
      dec.update(trit_b[i], 1, 3);
    }

    int root_b[2*NB+1];
    FOR(i, 2*NB+1) root_b[i] = -1;
    FOR(i, 2*NB+1) ;
    FOR(i, 2*NB+1) {
      root_b[i] = i;
      if(trit_b[i] == 0) {
        while(trit_b[root_b[i]] != 1) root_b[i] += 1;
      }else if(trit_b[i] == 2) {
        while(trit_b[root_b[i]] != 1) root_b[i] -= 1;
      }
    }
    
    int count_b[2*NB+1];
    memset(count_b, 0, sizeof(count_b));
    
    int max_b = dec.get(1<<16);
    dec.update(max_b, 1, (1<<16));
    FOR(i, 2*NB+1) if(i == root_b[i]) {
      count_b[i] = dec.get(max_b+1);
      dec.update(count_b[i], 1, max_b+1);
      count_b[i] = count_b[i]*count_b[i];
    }
    
    int cdf_b[2*NB+2];
    cdf_b[0] = 0;
    FOR(i, 2*NB+1) cdf_b[i+1] = cdf_b[i] + count_b[i];
    auto decode_b = [&]() -> int {
      int v = dec.get(cdf_b[2*NB+1]);
      int x = 0;
      while(v >= cdf_b[x+1]) x += 1;
      dec.update(cdf_b[x], count_b[x], cdf_b[2*NB+1]);
      return x - NB;
    };

    FOR(i, 2*L1) FOR(o, L2) {
      pnet.L1Weights[i][o] = 1.0 * decode_b() / NetworkQB;
    }

    auto decode_f = [&]() -> float {
      int y = dec.get(1<<15);
      dec.update(y, 1, 1<<15);
      float x = (1.0 * y / (1<<13)) - 2.0;
      return x;
    };

    FOR(o, L2) {
      pnet.L1Biases[o] = decode_f();
    }
    
    for(int i = 0; i < L2; ++i) 
      for(int o = 0; o < L3; ++o) 
        pnet.L2Weights[i][o] = decode_f();
    for(int o = 0; o < L3; ++o) 
      pnet.L2Biases[o] = decode_f();
    
    for(int i = 0; i < L3; ++i) 
      for(int b = 0; b < OutputBuckets; ++b) 
        pnet.L3Weights[i][b] = decode_f();
    for(int b = 0; b < OutputBuckets; ++b) 
      pnet.L3Biases[b] = decode_f();
  }
#endif

  void loadWeights() {
    // std::cout << "loading" << std::endl;
#ifndef SHARED
    std::cout << sizeof(PackedNet) << std::endl;
    std::cout << sizeof(Net) << std::endl;
#endif

#ifdef SAVESAMPLES
    SAMPLES.open("SAMPLES");
#endif

    PackedNet pnet;
#ifdef USE_QUANT
    load_packed_quant(pnet);
#else
    memcpy((char*) &pnet, gEmbeddedNNUEData, sizeof(PackedNet));
#endif

    // Weights = (Net*) Util::allocAlign(sizeof(Net));
    from_packed(Weights, pnet);
    // memcpy(Weights, gEmbeddedNNUEData, sizeof(Net));

    // Init NNZ table
    memset(nnzTable, 0, sizeof(nnzTable));
    for (int i = 0; i < 256; i++) {
      int j = 0;
      Bitboard bits = i;
      while (bits)
        nnzTable[i][j++] = popLsb(bits);
    }
   
    // dpbusd preprocessing:
    // done at quantisation time
    
    // Transpose weights so that we don't need to permute after packus, because
    // it interleaves each 128 block from a and each 128 block from b, alternately.
    // Instead we want it to concatenate a and b
    
    constexpr int weightsPerBlock = sizeof(__m128i) / sizeof(int16_t);
    constexpr int NumRegs = sizeof(VecI) / 8;
    __m128i regs[NumRegs];

    __m128i* ftWeights = (__m128i*) Weights.FeatureWeights;
    __m128i* ftBiases = (__m128i*) Weights.FeatureBiases;

    for (int i = 0; i < 768 * L1 / weightsPerBlock; i += NumRegs) {
      for (int j = 0; j < NumRegs; j++)
            regs[j] = ftWeights[i + j];

        for (int j = 0; j < NumRegs; j++)
            ftWeights[i + j] = regs[PackusOrder[j]];
    }

    for (int i = 0; i < L1 / weightsPerBlock; i += NumRegs) {
      for (int j = 0; j < NumRegs; j++)
            regs[j] = ftBiases[i + j];

        for (int j = 0; j < NumRegs; j++)
            ftBiases[i + j] = regs[PackusOrder[j]];
    }
  }
}
