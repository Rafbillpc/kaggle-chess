#include "header.hpp"
#include "range_coding.hpp"
#include <fstream>

f32 sigmoid(f32 x) {
  return 1.0 / (1.0 + exp(-x));
}

constexpr int SHIFT_UP_TO = 0;
constexpr int NUM_SHIFTS = 2*SHIFT_UP_TO+1;

constexpr int FeaturesWidth = 768;
constexpr int HL = 128;
constexpr int L1 = HL * NUM_SHIFTS;
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
  f32 FeatureWeights[768][L1];
  f32 FeatureBiases[L1];

  f32 L1Weights[L1*2][L2];
  f32 L1Biases[L2];

  f32 L2Weights[L2][L3];
  f32 L2Biases[L3];

  f32 L3Weights[OutputBuckets][L3];
  f32 L3Biases[OutputBuckets];
};

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
            for(int shifti = 0; shifti < NUM_SHIFTS; ++shifti) 
              {
                int shift = shifti - SHIFT_UP_TO;

                int sq = unmap_square(c+32*m);
                int x = sq/8;
                int y = sq%8;

                y -= shift;

                if(0 <= y && y < 8) {
                  net.FeatureWeights[(a*6+b)*64+8*x+y][shifti * HL + shuffle_2(o,m)]
                    = pnet.FeatureWeights[c][a][b][o];
                }
              }

  for(int o = 0; o < L1; ++o) 
    net.FeatureBiases[o] = pnet.FeatureBiases[o];

  for(int o = 0; o < L2; ++o) {
    for(int i = 0; i < 2*L1; ++i) {
      net.L1Weights[i][o] = pnet.L1Weights[i][o];
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

PackedNet pnet;
Net net;

struct sample_t {
  array<f32, 2*FeaturesWidth> features;
  int outputBucket;
};

int count[FeaturesWidth];
vector<sample_t> SAMPLES;
vector<float> SAMPLE_EVAL;

void load_samples(string const& filename, f32 prob) {
  ifstream is(filename);
  while(1) {
    int16_t nfeatures = 0;
    int16_t features[64];
    int32_t bucket;
    is.read((char*)&nfeatures, sizeof(int16_t));
    if(nfeatures == 0) break;
    is.read((char*)features, nfeatures * sizeof(int16_t));
    is.read((char*)&bucket, sizeof(int));
    bool force_keep = 0;
    // FOR(i, nfeatures) if(::count[features[i] % FeaturesWidth] == 0) {
    //   if(rng.randomDouble() < 20*prob) {
    //     force_keep = 1;
    //   }
    // }
    if(force_keep || rng.randomDouble() < prob) {
      sample_t S;
      FOR(i, 2*FeaturesWidth) S.features[i] = 0.0;
      FOR(i, nfeatures) S.features[features[i]] += 1.0;
      // FOR(i, nfeatures) debug(features[i] / 64 / 12,
      //                         features[i] / 64 % 12,
      //                         features[i] % 64);
      FOR(i, nfeatures) ::count[features[i] % FeaturesWidth] += 1;
      S.outputBucket = bucket;
      SAMPLES.pb(S);
    }
  }
}

f32 eval(Net const& net, sample_t const& s) {
  float ftOut[L1*2];
  float l1Out[L2];
  float l2Out[L3];
  float out;
  
  FOR(side, 2) FOR(o, L1) ftOut[side*L1+o] = net.FeatureBiases[o];
  FOR(side, 2) FOR(i,768) FOR(o, L1) {
    ftOut[side*L1+o] +=
      net.FeatureWeights[i][o] * s.features[side*768 + i];
  }
  FOR(o, L1*2) ftOut[o] = pow(clamp<f32>(ftOut[o], 0.0, 1.0), 2.0);
 
  // FOR(side, 2) {
  //   FOR(o, L1) {
  //     cout << ftOut[side*L1+o] << ' ';
  //   }
  //   cout << endl;
  // }
  
  FOR(o, L2) l1Out[o] = net.L1Biases[o];
  FOR(i, L1*2) FOR(o, L2) {
    l1Out[o] += net.L1Weights[i][o] * ftOut[i];
  }
  FOR(o, L2) l1Out[o] = pow(clamp<f32>(l1Out[o], 0.0, 1.0), 2.0);
  
  // FOR(o, L2) {
  //   cout << l1Out[o] << ' ';
  // }
  // cout << endl;
 
  FOR(o, L3) l2Out[o] = net.L2Biases[o];
  FOR(i, L2) FOR(o, L3) {
    l2Out[o] += net.L2Weights[i][o] * l1Out[i];
  }
  FOR(o, L3) l2Out[o] = clamp<f32>(l2Out[o], 0.0, 1.0);

  // FOR(o, L3) {
  //   cout << l2Out[o] << ' ';
  // }
  // cout << endl;

  // debug(s.outputBucket);

  // FOR(b, 8) {
  //   float out = net.L3Biases[b];
  //   FOR(i, L3) out += net.L3Weights[b][i] * l2Out[i];
  //   debug(b, out);
  // }


  out = net.L3Biases[s.outputBucket];
  FOR(i, L3) out += net.L3Weights[s.outputBucket][i] * l2Out[i];

  return out;
}

int16_t quant_a(float x) {
  float scaled = x * (float)NetworkQA;
  runtime_assert((int)abs(std::round(scaled)) <= 2*NetworkQA);
  return std::round(scaled);
}

int8_t quant_b(float x) {
  float scaled = x * (float)NetworkQB;
  return clamp<int>(std::round(scaled), -NetworkQB+1, NetworkQB-1);
}

void load_packed(string const& filename) {
  ifstream is(filename);
  is.read((char*)&pnet, sizeof(PackedNet));
}

PackedNet pqnet;
Net qnet;

const int NA = NetworkQA*2;
const int NB = NetworkQB;

struct state {

  int root_a[2*NA+1];
  array<i32, 2> range_a[2*NA+1];
  int root_b[2*NB+1];
  array<i32, 2> range_b[2*NA+1];

  f32 score;
  
  void reset() {
    FOR(i, 2*NA+1) {
      root_a[i] = i;
      range_a[i] = {i,i};
    }
    FOR(i, 2*NB+1) {
      root_b[i] = i;
      range_b[i] = {i,i};
    }

    score = calc_score();
  }
  
  void quantize() {
    pqnet = pnet;
    FOR(a, 32) FOR(b, 2) FOR(c, 6) FOR(d, HL) {
      int x = quant_a(pnet.FeatureWeights[a][b][c][d]);
      x = root_a[x + NA] - NA;
      pqnet.FeatureWeights[a][b][c][d]
        = 1.0 * x / NetworkQA;
    }
    FOR(o, L1) {
      int x = quant_a(pnet.FeatureBiases[o]);
      x = root_a[x + NA] - NA;
      pqnet.FeatureBiases[o]
        = 1.0 * x / NetworkQA;
    }
    FOR(i, 2*L1) FOR(o, L2) {
      int x = quant_b(pnet.L1Weights[i][o]);
      x = root_b[x + NB] - NB;
      pqnet.L1Weights[i][o]
        = 1.0 * x / NetworkQB;
    }

    from_packed(qnet, pqnet);
  }

  f32 eval_error() {
    quantize();

    f32 total_error = 0;

#pragma omp parallel for reduction(+:total_error)
    FOR(i, SAMPLES.size()) {
      f32 value = ::eval(qnet, SAMPLES[i]);
      f32 error = pow(abs(sigmoid(SAMPLE_EVAL[i]) - sigmoid(value)), 2.0);
      // f32 error = pow(SAMPLE_EVAL[i] - value, 2.0);
      total_error += error;
    }

    f32 avg_error = total_error / SAMPLES.size();
    return avg_error;
  }


  int write_quant(string const& filename = "") {
    static u8 buffer[10'000'000];
    range_encoder enc(buffer);
    
    vector<int> count_a(2*NA+1);
    FOR(a, 32) FOR(b, 2) FOR(c, 6) FOR(d, HL) {
      int x = quant_a(pnet.FeatureWeights[a][b][c][d]);
      count_a[root_a[x + NA]] += 1;
    }
    FOR(o, L1) {
      int x = quant_a(pnet.FeatureBiases[o]);
      count_a[root_a[x + NA]] += 1;
    }
    FOR(i, 2*NA+1) if(count_a[i] > 0) {
      count_a[i] = (int)sqrt(count_a[i]);
    }
    
    FOR(i, 2*NA+1) {
      int trit;
      if(i < root_a[i]) trit = 0;
      else if(i == root_a[i]) trit = 1;
      else trit = 2;
      enc.put(trit, 1, 3);
    }

    int max_a = *max_element(all(count_a));
    // debug(max_a);
    runtime_assert(max_a < (1<<16));
    enc.put(max_a, 1, (1<<16));
    FOR(i, 2*NA+1) if(i == root_a[i]) {
      enc.put(count_a[i], 1, max_a+1);
    }

    FOR(i, 2*NA+1) if(count_a[i] > 0) {
      count_a[i] = count_a[i]*count_a[i];
    }

    vector<int> cdf_a(2*NA+2);
    FOR(i, 2*NA+1) cdf_a[i+1] = cdf_a[i] + count_a[i];
    auto encode_a = [&](int x){
      x = root_a[x+NA];
      runtime_assert(0 <= x && x < 2*NA+1);
      enc.put(cdf_a[x], count_a[x], cdf_a[2*NA+1]);
    };
    
    FOR(a, 32) FOR(b, 2) FOR(c, 6) FOR(d, HL) {
      int x = quant_a(pnet.FeatureWeights[a][b][c][d]);
      encode_a(x);
    }
    FOR(o, L1) {
      int x = quant_a(pnet.FeatureBiases[o]);
      encode_a(x);
    }

    vector<int> count_b(2*NB+1);
    FOR(i, 2*L1) FOR(o, L2) {
      int x = quant_b(pnet.L1Weights[i][o]);
      count_b[root_b[x + NB]] += 1;
    }
    FOR(i, 2*NB+1) if(count_b[i] > 0) {
      count_b[i] = (int)sqrt(count_b[i]);
    }

    FOR(i, 2*NB+1) {
      int trit;
      if(i < root_b[i]) trit = 0;
      else if(i == root_b[i]) trit = 1;
      else trit = 2;
      enc.put(trit, 1, 3);
    }
    
    int max_b = *max_element(all(count_b));
    // debug(max_b);
    runtime_assert(max_b < (1<<16));
    enc.put(max_b, 1, (1<<16));
    FOR(i, 2*NB+1) if(i == root_b[i]) {
      enc.put(count_b[i], 1, max_b+1);
    }
    FOR(i, 2*NB+1) if(count_b[i] > 0) {
      count_b[i] = count_b[i]*count_b[i];
    }

    vector<int> cdf_b(2*NB+2);
    FOR(i, 2*NB+1) cdf_b[i+1] = cdf_b[i] + count_b[i];
    auto encode_b = [&](int x){
      x = root_b[x+NB];
      runtime_assert(0 <= x && x < 2*NB+1);
      enc.put(cdf_b[x], count_b[x], cdf_b[2*NB+1]);
    };

    FOR(i, 2*L1) FOR(o, L2) {
      int x = quant_b(pnet.L1Weights[i][o]);
      encode_b(x);
    }

    auto encode_f = [&](float x) {
      int y = (x + 2.0) * (1<<13);
      runtime_assert(0 <= y && y < (1<<15));
      enc.put(y, 1, (1<<15));
    };
    
    FOR(o, L2) {
      encode_f(pnet.L1Biases[o]);
    }

    for(int i = 0; i < L2; ++i) 
      for(int o = 0; o < L3; ++o) 
        encode_f(pnet.L2Weights[i][o]);
    for(int o = 0; o < L3; ++o) 
      encode_f(pnet.L2Biases[o]);
    
    for(int i = 0; i < L3; ++i) 
      for(int b = 0; b < OutputBuckets; ++b) 
        encode_f(pnet.L3Weights[i][b]);
    for(int b = 0; b < OutputBuckets; ++b) 
      encode_f(pnet.L3Biases[b]);

    enc.put(1, 1, 1<<16);
    enc.finish();
    // debug(enc.size);

    if(!filename.empty()) {
      ofstream os(filename);
      os.write((char*)buffer, enc.size);
    }

    return enc.size;
  }
  
  f32 eval_cost(bool true_cost = true) {
    return write_quant("");
  }
  
  f32 calc_score() {
    f32 a = eval_error();
    f32 a0 = a - 1.4e-4;
    a0 = max(a0, 0.f);
    f32 b = eval_cost();
    b -= 40000;
    b = max(b, 0.f);
    return 1e12 * a0 + b + a;
  }

  void set_root_a(int i, int l, int r) {
    range_a[i] = {l,r};
    FORU(x, l, r) root_a[x] = i;
  }
 
  void set_root_b(int i, int l, int r) {
    range_b[i] = {l,r};
    FORU(x, l, r) root_b[x] = i;
  }

  void transition0(f32 temp) {
    int i = rng.random32(2*NA);
    int ra = root_a[i], rb = root_a[i+1];
    if(ra == rb) {
      auto [l1, r2] = range_a[ra];
      int r1 = i, l2 = i+1;

      int new_root_a = rng.randomRange32(l1, r1);
      int new_root_b = rng.randomRange32(l2, r2);
      
      set_root_a(new_root_a, l1, r1);
      set_root_a(new_root_b, l2, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_a(ra, l1, r2);
      }
    }else{
      auto [l1, r1] = range_a[ra];
      auto [l2, r2] = range_a[rb];

      int new_root = rng.randomRange32(l1, r2);

      set_root_a(new_root, l1, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_a(ra, l1, r1);
        set_root_a(rb, l2, r2);
      }
    }
  }

  void transition1(f32 temp) {
    int i = rng.random32(2*NA);
    int ra = root_a[i], rb = root_a[i+1];
    if(ra == rb) return;
    i32 dir = rng.random32(2);
    if(dir == 0 && ra != i) {
      auto [l1, r1] = range_a[ra];
      auto [l2, r2] = range_a[rb];

      set_root_a(ra, l1, r1-1);
      set_root_a(rb, l2-1, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_a(ra, l1, r1);
        set_root_a(rb, l2, r2);
      }
    }else if(dir == 1 && rb != i+1) {
      auto [l1, r1] = range_a[ra];
      auto [l2, r2] = range_a[rb];

      set_root_a(ra, l1, r1+1);
      set_root_a(rb, l2+1, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_a(ra, l1, r1);
        set_root_a(rb, l2, r2);
      }
    }
  }

  void transition4(f32 temp) {
    int i = root_a[rng.random32(2*NA+1)];
    auto [l,r] = range_a[i];
    int new_root = rng.randomRange32(l,r);

    set_root_a(new_root, l, r);
    
    f32 new_score = calc_score();
    if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
      score = new_score;
    }else{
      set_root_a(i, l, r);
    }
  }

  void transition2(f32 temp) {
    int i = rng.random32(2*NB);
    int ra = root_b[i], rb = root_b[i+1];
    if(ra == rb) {
      auto [l1, r2] = range_b[ra];
      int r1 = i, l2 = i+1;

      int new_root_a = rng.randomRange32(l1, r1);
      int new_root_b = rng.randomRange32(l2, r2);
      
      set_root_b(new_root_a, l1, r1);
      set_root_b(new_root_b, l2, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_b(ra, l1, r2);
      }
    }else{
      auto [l1, r1] = range_b[ra];
      auto [l2, r2] = range_b[rb];

      int new_root = rng.randomRange32(l1, r2);

      set_root_b(new_root, l1, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_b(ra, l1, r1);
        set_root_b(rb, l2, r2);
      }
    }
  }

  void transition3(f32 temp) {
    int i = rng.random32(2*NB);
    int ra = root_b[i], rb = root_b[i+1];
    if(ra == rb) return;
    i32 dir = rng.random32(2);
    if(dir == 0 && ra != i) {
      auto [l1, r1] = range_b[ra];
      auto [l2, r2] = range_b[rb];

      set_root_b(ra, l1, r1-1);
      set_root_b(rb, l2-1, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_b(ra, l1, r1);
        set_root_b(rb, l2, r2);
      }
    }else if(dir == 1 && rb != i+1) {
      auto [l1, r1] = range_b[ra];
      auto [l2, r2] = range_b[rb];

      set_root_b(ra, l1, r1+1);
      set_root_b(rb, l2+1, r2);

      f32 new_score = calc_score();
      if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
        score = new_score;
      }else{
        set_root_b(ra, l1, r1);
        set_root_b(rb, l2, r2);
      }
    }
  }

  void transition5(f32 temp) {
    int i = root_b[rng.random32(2*NB+1)];
    auto [l,r] = range_b[i];
    int new_root = rng.randomRange32(l,r);

    set_root_b(new_root, l, r);
    
    f32 new_score = calc_score();
    if(new_score < score || new_score-score <= temp * rng.randomDouble()) {
      score = new_score;
    }else{
      set_root_b(i, l, r);
    }
  }
  
  void transition(f32 temp) {
    i32 ty = rng.random32(6);
    if(ty == 0) transition0(temp);
    if(ty == 1) transition1(temp);
    if(ty == 2) transition2(temp);
    if(ty == 3) transition3(temp);
    if(ty == 4) transition4(temp);
    if(ty == 5) transition5(temp);

    // check();
  }

  void check() {
    FOR(i, 2*NA+1) {
      runtime_assert(root_a[root_a[i]] == root_a[i]);
    }
    FOR(i, 2*NA+1) if(root_a[i] == i) {
      auto [l,r] = range_a[i];
      runtime_assert(l <= i && i <= r);
      FORU(x, l, r) runtime_assert(root_a[x] == i);
    }
  }
  
};

int main() {
  // rng.reset(42);
  debug(sizeof(PackedNet));
  load_packed("nn.net");
  from_packed(net, pnet);
  load_samples("SAMPLES", 0.001);
  debug(SAMPLES.size());

  SAMPLE_EVAL.resize(SAMPLES.size());
  FOR(i, SAMPLES.size()) {
    SAMPLE_EVAL[i] = eval(net, SAMPLES[i]);
  }

  // vector<f32> total(12, 0.0);
  // FOR(a, 32) FOR(b, 2) FOR(c, 6) FOR(d, HL) {
  //   total[b*6+c] += pnet.FeatureWeights[a][b][c][d];
  // }
  // FOR(i, 12) total[i] /= (32*HL);
  // debug(total);
  
  state S;
  S.reset();
  debug(S.score);
  debug(S.eval_error());
  debug(S.eval_cost());
  S.write_quant("nn_quant.net");
  
  // S.write_quant("nn_quant.net");exit(0);
  
  f32 best_score = S.score;

  f32 temp0 = 1e-9;
  f32 temp1 = 1e-9;
 
  const int NUM_ITERS = 100'000;
  int niter = 0;
  while(1) {
    niter += 1;
    f32 done = 1.0 * niter / NUM_ITERS;
    f32 temp = temp0*pow(temp1/temp0, done);
    if(niter % 100 == 0){
      cerr 
        << "iter = " << niter 
        << ", score = " << S.score 
        << ", best = " << best_score
        << ", error = " << S.eval_error()
        << ", cost = " << S.eval_cost()
        << ", temp = " << temp
        << endl;

      int size = S.write_quant("nn_quant.net");
      debug(size);
    }

    S.transition(temp);
    if(S.score < best_score) {
      best_score = S.score;
    }
  }
  
  return 0;
}

/*
 * TODO: two options for every weight (round either up or down ?)
 */
