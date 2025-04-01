// Obsidian.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "nnue.h"
#include "search.h"
#include "threads.h"
#include "movegen.h"
#include "cuckoo.h"
#include "history.h"
#include "threads.h"
#include "tt.h"
#include "types.h"
#include "uci.h"

#include <ctime>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/syscall.h>

timespec clock_gettime() {
  int ret;
  clockid_t clk_id = CLOCK_MONOTONIC;
  timespec tp;

  asm volatile
    (
     "syscall"
     : "=a" (ret)
       //                 EDI      RSI       RDX
     : "0"(SYS_clock_gettime), "D"(clk_id), "S"(&tp)
     : "rcx", "r11", "memory"
     );

  return tp;
}

timespec time_start;

void init_time() {
  time_start = clock_gettime();
}

int64_t timeMillis() {
  auto time_now = clock_gettime();
  long long int s = time_now.tv_sec - time_start.tv_sec;
  long long int ns = time_now.tv_nsec - time_start.tv_nsec;

  long long int m = (s * 1'000'000'000 + ns) / 1'000'000;
 
  return m;
}

#ifdef SHARED

__attribute__((visibility("default")))
extern "C"
void init() {
  init_time();

  Zobrist::init();

  Bitboards::init();

  positionInit();

  Cuckoo::init();

  Search::init();

  // UCI::init();

  Threads::setThreadCount();
  TT::resize(1); // UCI::Options["Hash"]);

  NNUE::loadWeights();

  // UCI::loop(argc, argv);

  // Threads::setThreadCount(0);
}

Position pos;
Search::PrevPositions prevPositions;

void uci_squareToString(char* buf, int& len, Square s) {
  buf[len++] = char('a' + fileOf(s));
  buf[len++] = char('1' + rankOf(s));
}

void uci_moveToString(char* buf, int& len, Move m) {
  len = 0;
  
  if (m == MOVE_NONE)
    return;

  uci_squareToString(buf, len, move_from(m));
  uci_squareToString(buf, len, move_to(m));

  if (move_type(m) == MT_PROMOTION)
    buf[len++] = "  nbrq"[promo_type(m)];
}

char tolower(char c) {
  if('A' <= c && c <= 'Z') return c-'A'+'a';
  return c;
}

Move uci_stringToMove(const Position& pos, char* str) {
  int strlen = 0;
  while(str[strlen] != ' ') strlen += 1;

  if(strlen == 5)
    str[4] = char(tolower(str[4]));

  MoveList moves;
  getStageMoves(pos, ADD_ALL_MOVES, &moves);

  for (const auto& m : moves) {
    char buf[5];
    int buflen;
    uci_moveToString(buf, buflen, m.move);
    if(buflen == strlen) {
      bool ok = 1;
      for(int i = 0; i < buflen; ++i) if(buf[i] != str[i]) ok = 0;
      if(ok) 
        return m.move;
    }
  }
  return MOVE_NONE;
}

__attribute__((visibility("default")))
extern "C"
void position(char const* fen, char const* moves) {
  pos.setToFen(fen);
  prevPositions.clear();
  prevPositions.push(pos.key);

  int imove = 0;
  while(moves[imove] != 0) {
    Move m = uci_stringToMove(pos, const_cast<char*>(moves + imove));

    DirtyPieces dirtyPieces;
    pos.doMove(m, dirtyPieces);

    // If this move reset the half move clock, we can ignore and forget all the previous position
    if (pos.halfMoveClock == 0)
      prevPositions.clear();

    prevPositions.push(pos.key);
    
    while(moves[imove] != ' ') imove += 1;
    imove += 1;
  }
  
  // Remove the last position because it is equal to the current position
  prevPositions.count -= 1;
}

__attribute__((visibility("default")))
extern "C"
char const* go(int wtime, int btime, int winc, int binc) {
  Search::Settings searchSettings;
  searchSettings.startTime = timeMillis();
  searchSettings.position = pos;
  searchSettings.prevPositions = prevPositions;

  searchSettings.time[WHITE] = wtime;
  searchSettings.time[BLACK] = btime;
  searchSettings.inc[WHITE] = winc;
  searchSettings.inc[BLACK] = binc;

  TT::nextSearch();
  Threads::startSearch(searchSettings);

  Move m = Threads::searchThread.rootMoves[0].move;
  static char buf[6];
  int buflen = 0;
  uci_moveToString(buf, buflen, m);
  buf[buflen] = 0;
  return buf;
}

#else

int main(int argc, char** argv)
{
  init_time();
  std::cout << "Obsidian " << engineVersion << " by Gabriele Lombardo" << std::endl;

  // std::cout << sizeof(MainHistory) << std::endl;
  // std::cout << sizeof(CaptureHistory) << std::endl;
  // std::cout << sizeof(ContinuationHistory) << std::endl;
  // std::cout << sizeof(CounterMoveHistory) << std::endl;
  // std::cout << sizeof(PawnCorrHist) << std::endl;
  // std::cout << sizeof(NonPawnCorrHist) << std::endl;

  
  // std::cout << sizeof(NNUE::Accumulator)*MAX_PLY << std::endl;
  // std::cout << sizeof(Search::SearchInfo)*(MAX_PLY + Search::SsOffset) << std::endl;
  
  Zobrist::init();

  Bitboards::init();

  positionInit();

  Cuckoo::init();

  Search::init();

  UCI::init();

  Threads::setThreadCount();
  TT::resize(UCI::Options["Hash"]);

  NNUE::loadWeights();

  UCI::loop(argc, argv);

  return 0;
}

#endif
