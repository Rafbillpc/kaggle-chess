#pragma once

#include "history.h"
#include "nnue.h"
#include "position.h"
#include "types.h"

#ifndef SHARED
#include <iostream>
#include <string>
#include <sstream>
#endif

namespace Search {

  struct PrevPositions {
    int count;
    uint64_t data[512];

    void clear() {
      count = 0;
    }

    void push(uint64_t x) {
      data[count++] = x;
    }

    uint64_t const& operator[](int ix) const { return data[ix]; }
  };
  
  struct Settings {

    int64_t time[COLOR_NB], inc[COLOR_NB], movetime, startTime;
    int movestogo, depth;
    uint64_t nodes;

    Position position;

    PrevPositions prevPositions;

    Settings();

    inline bool standardTimeLimit() const {
      return time[WHITE] || time[BLACK];
    }
  };

  struct SearchInfo {
    Score staticEval;
    Move playedMove;
    bool playedCap;

    Move killerMove;

    Move pv[MAX_PLY];
    int pvLength;

    // [piece to]
    ContinuationHistory_at contHistory;
  };

  // A sort of header of the search stack, so that plies behind 0 are accessible and
  // it's easier to determine conthist score, improving, ...
  constexpr int SsOffset = 6;

  class Thread {

  public:
    volatile bool searching = false;
    volatile bool exitThread = false;

    volatile int completeDepth;
    volatile uint64_t nodesSearched;
    volatile uint64_t tbHits;

    Thread();

    void resetHistories();

    int64_t optimumTime, maxTime;
    uint32_t maxTimeCounter;

    int rootDepth;

    int ply = 0;

    int keyStackHead;
    Key keyStack[100 + MAX_PLY];

    int accumStackHead;
    NNUE::Accumulator accumStack[MAX_PLY];

    SearchInfo searchStack[MAX_PLY + SsOffset];

    RootMoveList rootMoves;
    int pvIdx;

    MainHistory mainHistory;
    CaptureHistory captureHistory;
    ContinuationHistory contHistory;
    CounterMoveHistory counterMoveHistory;
    PawnCorrHist pawnCorrhist;
    NonPawnCorrHist wNonPawnCorrhist;
    NonPawnCorrHist bNonPawnCorrhist;

    NNUE::FinnyTable finny;

    Score searchPrevScore;

    void refreshAccumulator(Position& pos, NNUE::Accumulator& acc, Color side);

    void updateAccumulator(Position& pos, NNUE::Accumulator& acc);

    Score doEvaluation(Position& position);

    void sortRootMoves(int offset);

    bool visitRootMove(Move move);

    void playNullMove(Position& pos, SearchInfo* ss);

    void cancelNullMove();

    void playMove(Position& pos, Move move, SearchInfo* ss);

    void cancelMove();

    int getQuietHistory(Position& pos, Move move, SearchInfo* ss);

    int getCapHistory(Position& pos, Move move);

    // Perform adjustments such as 50MR, correction history, contempt, ...
    int adjustEval(Position &pos, Score staticEval);

    void updateHistories(Position& pos, int bonus, int malus, Move bestMove,
      Move* quietMoves, int quietCount, int depth, SearchInfo* ss);

    bool hasUpcomingRepetition(Position& pos, int ply);

    // Should not be called from Root node
    bool isRepetition(Position& pos, int ply);

    Score qsearch(bool IsPV, Position& position, Score alpha, Score beta, int depth, SearchInfo* ss);

    Score negamax(bool IsPV, Position& position, Score alpha, Score beta, int depth,
      bool cutNode, SearchInfo* ss, const Move excludedMove = MOVE_NONE);

    void startSearch();
  };

  template<bool root>
  int64_t perft(Position& pos, int depth);

  void initLmrTable();

  void init();

#ifndef SHARED
  void printInfo(int depth, int pvIdx, Score score, const std::string& pvString);
#endif
}
