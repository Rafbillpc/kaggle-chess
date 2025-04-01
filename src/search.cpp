#include "search.h"
#include "cuckoo.h"
#include "evaluate.h"
#include "movepick.h"
#include "timeman.h"
#include "threads.h"
#include "tt.h"
#include "tuning.h"
#include "uci.h"
#include "zobrist.h"

#include <climits>

namespace Search {

  DEFINE_PARAM_S(QsFpMargin, 143, 14);

  DEFINE_PARAM_S(LmrBase, 100, 10);
  DEFINE_PARAM_S(LmrDiv, 287, 29);

  DEFINE_PARAM_S(PawnChWeight, 40, 5);
  DEFINE_PARAM_S(NonPawnChWeight, 50, 5);

  DEFINE_PARAM_S(StatBonusBias, -23, 50);
  DEFINE_PARAM_S(StatBonusLinear, 160, 15);
  DEFINE_PARAM_S(StatBonusMax, 1277, 100);
  DEFINE_PARAM_S(StatBonusBoostAt, 111, 10);

  DEFINE_PARAM_S(StatMalusBias, 17, 50);
  DEFINE_PARAM_S(StatMalusLinear, 186, 15);
  DEFINE_PARAM_S(StatMalusMax, 1127, 100);

  DEFINE_PARAM_S(EvalHistA, 58, 6);
  DEFINE_PARAM_S(EvalHistB, -494, 45);
  DEFINE_PARAM_S(EvalHistC, 509, 50);

  DEFINE_PARAM_S(RazoringDepthMul, 355, 38);

  DEFINE_PARAM_S(RfpMaxDepth, 11, 1);
  DEFINE_PARAM_S(RfpDepthMul, 85, 8);

  DEFINE_PARAM_S(NmpBase, 4, 1);
  DEFINE_PARAM_B(NmpDepthDiv, 3, 1, 21);
  DEFINE_PARAM_S(NmpEvalDiv, 151, 15);
  DEFINE_PARAM_S(NmpEvalDivMin, 4, 1);
  DEFINE_PARAM_S(NmpA, 27, 3);
  DEFINE_PARAM_S(NmpB, 205, 20);

  DEFINE_PARAM_S(ProbcutBetaMargin, 176, 18);

  DEFINE_PARAM_S(HistPrDepthMul, -4845, 460);

  DEFINE_PARAM_S(LmpBase,    3, 1);

  DEFINE_PARAM_S(QsSeeMargin, -32, 15);
  DEFINE_PARAM_S(PvsQuietSeeMargin, -35, 10);
  DEFINE_PARAM_S(PvsCapSeeMargin, -95, 10);

  DEFINE_PARAM_S(EarlyLmrHistoryDiv, 3489, 360);

  DEFINE_PARAM_S(FpBase, 160, 17);
  DEFINE_PARAM_S(FpMaxDepth, 10, 1);
  DEFINE_PARAM_S(FpDepthMul, 134, 13);

  DEFINE_PARAM_S(SBetaMargin, 64, 6);
  DEFINE_PARAM_S(TripleExtMargin, 121, 13);
  DEFINE_PARAM_S(DoubleExtMargin, 13, 1);

  DEFINE_PARAM_S(LmrQuietHistoryDiv, 8794, 860);
  DEFINE_PARAM_S(LmrCapHistoryDiv, 6319, 660);
  DEFINE_PARAM_S(ZwsDeeperMargin, 78, 8);

  DEFINE_PARAM_B(AspWindowStartDepth, 4, 4, 34);
  DEFINE_PARAM_B(AspWindowStartDelta, 11, 5, 25);

  const float s_log_C0 = -19.645704f;
  const float s_log_C1 = 0.767002f;
  const float s_log_C2 = 0.3717479f;
  const float s_log_C3 = 5.2653985f;
  const float s_log_C4 = -(1.0f + s_log_C0) * (1.0f + s_log_C1) / ((1.0f + s_log_C2) * (1.0f + s_log_C3)); //ensures that log(1) == 0
  const float s_log_2 = 0.6931472f;

  // assumes x > 0 and that it's not a subnormal.
  // Results for 0 or negative x won't be -Infinity or NaN
  __attribute__((noinline))
  inline float fast_log(float x)
  {
    unsigned int ux = std::bit_cast<unsigned int>(x);
    int e = static_cast<int>(ux - 0x3f800000) >> 23; //e = exponent part can be negative
    ux |= 0x3f800000;
    ux &= 0x3fffffff; // 1 <= x < 2  after replacing the exponent field
    x = std::bit_cast<float>(ux);
    float a = (x + s_log_C0) * (x + s_log_C1);
    float b = (x + s_log_C2) * (x + s_log_C3);
    float c = (float(e) + s_log_C4);
    float d = a / b;
    return (c + d) * s_log_2;
  }
  
  // constexpr float logTable[224] =
  //   { 0.0,0.0,0.6931471805599453,1.0986122886681098,1.3862943611198906,1.6094379124341003,1.791759469228055,1.9459101490553132,2.0794415416798357,2.1972245773362196,2.302585092994046,2.3978952727983707,2.4849066497880004,2.5649493574615367,2.6390573296152584,2.70805020110221,2.772588722239781,2.833213344056216,2.8903717578961645,2.9444389791664403,2.995732273553991,3.044522437723423,3.091042453358316,3.1354942159291497,3.1780538303479458,3.2188758248682006,3.258096538021482,3.295836866004329,3.332204510175204,3.367295829986474,3.4011973816621555,3.4339872044851463,3.4657359027997265,3.4965075614664802,3.5263605246161616,3.5553480614894135,3.58351893845611,3.6109179126442243,3.6375861597263857,3.6635616461296463,3.6888794541139363,3.713572066704308,3.7376696182833684,3.7612001156935624,3.784189633918261,3.8066624897703196,3.828641396489095,3.8501476017100584,3.871201010907891,3.8918202981106265,3.912023005428146,3.9318256327243257,3.9512437185814275,3.970291913552122,3.9889840465642745,4.007333185232471,4.02535169073515,4.04305126783455,4.060443010546419,4.07753744390572,4.0943445622221,4.110873864173311,4.127134385045092,4.143134726391533,4.1588830833596715,4.174387269895637,4.189654742026425,4.204692619390966,4.219507705176107,4.23410650459726,4.248495242049359,4.2626798770413155,4.276666119016055,4.290459441148391,4.30406509320417,4.31748811353631,4.330733340286331,4.343805421853684,4.356708826689592,4.3694478524670215,4.382026634673881,4.394449154672439,4.406719247264253,4.418840607796598,4.430816798843313,4.442651256490317,4.454347296253507,4.465908118654584,4.477336814478207,4.48863636973214,4.499809670330265,4.51085950651685,4.5217885770490405,4.532599493153256,4.543294782270004,4.553876891600541,4.564348191467836,4.574710978503383,4.584967478670572,4.59511985013459,4.605170185988092,4.61512051684126,4.624972813284271,4.634728988229636,4.6443908991413725,4.653960350157523,4.663439094112067,4.672828834461906,4.68213122712422,4.6913478822291435,4.700480365792417,4.709530201312334,4.718498871295094,4.727387818712341,4.736198448394496,4.74493212836325,4.7535901911063645,4.762173934797756,4.770684624465665,4.77912349311153,4.787491742782046,4.795790545596741,4.804021044733257,4.812184355372417,4.820281565605037,4.8283137373023015,4.836281906951478,4.844187086458591,4.852030263919617,4.859812404361672,4.867534450455582,4.875197323201151,4.882801922586371,4.890349128221754,4.897839799950911,4.90527477843843,4.912654885736052,4.919980925828125,4.927253685157205,4.9344739331306915,4.941642422609304,4.948759890378168,4.955827057601261,4.962844630259907,4.969813299576001,4.976733742420574,4.983606621708336,4.990432586778736,4.997212273764115,5.003946305945459,5.0106352940962555,5.017279836814924,5.0238805208462765,5.030437921392435,5.0369526024136295,5.043425116919247,5.049856007249537,5.056245805348308,5.062595033026967,5.0689042022202315,5.075173815233827,5.081404364984463,5.087596335232384,5.093750200806762,5.099866427824199,5.10594547390058,5.111987788356544,5.117993812416755,5.123963979403259,5.1298987149230735,5.135798437050262,5.14166355650266,5.147494476813453,5.153291594497779,5.159055299214529,5.1647859739235145,5.170483995038151,5.176149732573829,5.181783550292085,5.187385805840755,5.19295685089021,5.198497031265826,5.204006687076795,5.209486152841421,5.214935757608986,5.220355825078324,5.225746673713202,5.231108616854587,5.236441962829949,5.241747015059643,5.247024072160486,5.25227342804663,5.2574953720277815,5.262690188904886,5.267858159063328,5.272999558563747,5.278114659230517,5.2832037287379885,5.288267030694535,5.293304824724492,5.298317366548036,5.303304908059076,5.308267697401205,5.313205979041787,5.318119993844216,5.3230099791384085,5.327876168789581,5.332718793265369,5.337538079701318,5.342334251964811,5.3471075307174685,5.351858133476067,5.356586274672012,5.3612921657094255,5.365976015021851,5.3706380281276624,5.375278407684165,5.37989735354046,5.384495062789089,5.389071729816501,5.393627546352362,5.3981627015177525,5.402677381872279,5.407171771460119
  //   };
  int lmrTable[MAX_PLY][MAX_MOVES];

  Settings::Settings() {
    time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = movetime = 0;
    movestogo = 0;
    depth = MAX_PLY-4; // no depth limit by default
    nodes = 0;
  }

  int pieceTo(Position& pos, Move m) {
    return pos.board[move_from(m)] * SQUARE_NB + move_to(m);
  }

  void initLmrTable() {
    // avoid log(0) because it's negative infinity
    lmrTable[0][0] = 0;

    double dBase = LmrBase / 100.0;
    double dDiv = LmrDiv / 100.0;

    for (int i = 1; i < MAX_PLY; i++) {
      for (int m = 1; m < MAX_MOVES; m++) {
        lmrTable[i][m] = dBase + fast_log(i) * fast_log(m) / dDiv;
      }
    }
  }

  // Called once at engine initialization
  void init() {
    initLmrTable();
  }

  void Thread::resetHistories() {
    memset(mainHistory, 0, sizeof(mainHistory));
    memset(captureHistory, 0, sizeof(captureHistory));
    memset(counterMoveHistory, 0, sizeof(counterMoveHistory));
    memset(contHistory, 0, sizeof(contHistory));
    memset(pawnCorrhist, 0, sizeof(pawnCorrhist));
    memset(wNonPawnCorrhist, 0, sizeof(wNonPawnCorrhist));
    memset(bNonPawnCorrhist, 0, sizeof(bNonPawnCorrhist));

    searchPrevScore = SCORE_NONE;
  }

  Thread::Thread()
  {
    resetHistories();
  }

#ifndef SHARED
  template<bool root>
  int64_t perft(Position& pos, int depth) {

    MoveList moves;
    getStageMoves(pos, ADD_ALL_MOVES, &moves);

    if (depth <= 1) {
      int n = 0;
      for (int i = 0; i < moves.size(); i++)
        n += pos.isLegal(moves[i].move);
      return n;
    }

    int64_t n = 0;
    for (int i = 0; i < moves.size(); i++) {
      Move move = moves[i].move;

      if (!pos.isLegal(move))
        continue;

      DirtyPieces dirtyPieces;

      Position newPos = pos;
      newPos.doMove(move, dirtyPieces);

      int64_t thisNodes = perft<false>(newPos, depth - 1);
      if constexpr (root)
        std::cout << UCI::moveToString(move) << " -> " << thisNodes << std::endl;

      n += thisNodes;
    }
    return n;
  }
  
  template int64_t perft<false>(Position&, int);
  template int64_t perft<true>(Position&, int);
#endif

  int64_t elapsedTime() {
    return timeMillis() - Threads::getSearchSettings().startTime;
  }

#ifndef SHARED
  void printInfo(int depth, int pvIdx, Score score, const std::string& pvString) {
    const int64_t elapsed = elapsedTime();
    std::ostringstream infoStr;
        infoStr
          << "info"
          << " depth "    << depth
          << " multipv "  << pvIdx
          << " score "    << UCI::scoreToString(score)
          << " nodes "    << Threads::totalNodes()
          << " nps "      << (Threads::totalNodes() * 1000ULL) / std::max<int64_t>(elapsed, 1LL)
          << " hashfull " << TT::hashfull()
          << " tbhits "   << Threads::totalTbHits()
          << " time "     << elapsed
          << " pv "       << pvString;

    std::cout << infoStr.str() << std::endl;
  }
#endif

  void printBestMove(Move move) {
#ifndef SHARED
    std::cout << "bestmove " << UCI::moveToString(move) << std::endl;
#endif
  }

  int statBonus(int d) {
    return std::min(StatBonusLinear * d + StatBonusBias, (int)StatBonusMax);
  }

  int statMalus(int d) {
    return std::min(StatMalusLinear * d + StatMalusBias, (int)StatMalusMax);
  }

  void Thread::sortRootMoves(int offset) {
    for (int i = offset; i < rootMoves.size(); i++) {
      int best = i;

      for (int j = i + 1; j < rootMoves.size(); j++)
        if (rootMoves[j].score > rootMoves[best].score)
          best = j;

      if (best != i)
        std::swap(rootMoves[i], rootMoves[best]);
    }
  }

  bool Thread::visitRootMove(Move move) {
    for (int i = pvIdx; i < rootMoves.size(); i++) {
      if (move == rootMoves[i].move)
        return true;
    }
    return false;
  }

  void Thread::playNullMove(Position& pos, SearchInfo* ss) {
    ss->contHistory = {
      .table = contHistory[false],
      .hash = 0,
    };
    ss->playedMove = MOVE_NONE;
    ss->playedCap = false;
    keyStack[keyStackHead++] = pos.key;

    ply++;
    pos.doNullMove();
  }

  void Thread::cancelNullMove() {
    ply--;
    keyStackHead--;
  }

  void Thread::refreshAccumulator(Position& pos, NNUE::Accumulator& acc, Color side) {
    const Square king = pos.kingSquare(side);
    NNUE::FinnyEntry& entry = finny[fileOf(king) >= FILE_E];

    for (Color c = WHITE; c <= BLACK; ++c) {
      for (PieceType pt = PAWN; pt <= KING; ++pt) {
        const Bitboard oldBB = entry.byColorBB[side][c] & entry.byPieceBB[side][pt];
        const Bitboard newBB = pos.pieces(c, pt);
        Bitboard toRemove = oldBB & ~newBB;
        Bitboard toAdd = newBB & ~oldBB;

        while (toRemove) {
          Square sq = popLsb(toRemove);
          entry.acc.removePiece(king, side, makePiece(c, pt), sq);
        }
        while (toAdd) {
          Square sq = popLsb(toAdd);
          entry.acc.addPiece(king, side, makePiece(c, pt), sq);
        }
      }
    }
    acc.updated[side] = true;
    memcpy(acc.colors[side], entry.acc.colors[side], sizeof(acc.colors[0]));
    memcpy(entry.byColorBB[side], pos.byColorBB, sizeof(entry.byColorBB[0]));
    memcpy(entry.byPieceBB[side], pos.byPieceBB, sizeof(entry.byPieceBB[0]));
  }

  void Thread::updateAccumulator(Position& pos, NNUE::Accumulator& head) {

    for (Color side = WHITE; side <= BLACK; ++side) {
      if (head.updated[side])
        continue;

      const Square king = head.kings[side];
      NNUE::Accumulator* iter = &head;
      while (true) {
        iter--;

        if (NNUE::needRefresh(side, iter->kings[side], king)) {
          refreshAccumulator(pos, head, side);
          break;
        }

        if (iter->updated[side]) {
          NNUE::Accumulator* lastUpdated = iter;
          while (lastUpdated != &head) {
            (lastUpdated+1)->doUpdates(king, side, *lastUpdated);
            lastUpdated++;
          }
          break;
        }
      }
    }
  }

  Score Thread::doEvaluation(Position& pos) {
    NNUE::Accumulator& acc = accumStack[accumStackHead];
    updateAccumulator(pos, acc);
    return Eval::evaluate(pos, !(ply % 2), acc);
  }

  void Thread::playMove(Position& pos, Move move, SearchInfo* ss) {

    nodesSearched++;

    const bool isCap = pos.board[move_to(move)] != NO_PIECE;
    ss->contHistory = {
      .table = contHistory[isCap],
      .hash = CH_PIECE1[pos.board[move_from(move)]] ^ CH_SQUARE1[move_to(move)]
    };
    ss->playedMove = move;
    ss->playedCap = ! pos.isQuiet(move);
    keyStack[keyStackHead++] = pos.key;

    NNUE::Accumulator& newAcc = accumStack[++accumStackHead];

    ply++;
    pos.doMove(move, newAcc.dirtyPieces);

    for (Color side = WHITE; side <= BLACK; ++side) {
      newAcc.updated[side] = false;
      newAcc.kings[side] = pos.kingSquare(side);
    }
  }

  void Thread::cancelMove() {
    ply--;
    keyStackHead--;
    accumStackHead--;
  }

  int Thread::getCapHistory(Position& pos, Move move) {
    PieceType captured = piece_type(pos.board[move_to(move)]);
    return captureHistory[pieceTo(pos, move)][captured];
  }

  int Thread::getQuietHistory(Position& pos, Move move, SearchInfo* ss) {
    uint32_t chIndex = CH_PIECE2[pos.board[move_from(move)]] ^ CH_SQUARE2[move_to(move)];
    return    mainHistory[pos.sideToMove][move_from_to(move)]
            + (ss - 1)->contHistory.at(chIndex)
            + (ss - 2)->contHistory.at(chIndex)
            + (ss - 4)->contHistory.at(chIndex);
  }

  Score Thread::adjustEval(Position &pos, Score eval) {
    // 50 move rule scaling
    eval = (eval * (200 - pos.halfMoveClock)) / 200;

    // Pawn correction history
    eval += PawnChWeight * pawnCorrhist[pos.sideToMove][getCorrHistIndex(pos.pawnKey)] / 512;
    eval += NonPawnChWeight * wNonPawnCorrhist[pos.sideToMove][getCorrHistIndex(pos.nonPawnKey[WHITE])] / 512;
    eval += NonPawnChWeight * bNonPawnCorrhist[pos.sideToMove][getCorrHistIndex(pos.nonPawnKey[BLACK])] / 512;

    return std::clamp(eval, SCORE_TB_LOSS_IN_MAX_PLY + 1, SCORE_TB_WIN_IN_MAX_PLY - 1);
  }

  void addToContHistory(Position& pos, int bonus, Move move, SearchInfo* ss) {
    uint32_t chIndex = CH_PIECE2[pos.board[move_from(move)]] ^ CH_SQUARE2[move_to(move)];
    if ((ss - 1)->playedMove)
      addToHistory((ss - 1)->contHistory.at(chIndex), bonus);
    if ((ss - 2)->playedMove)
      addToHistory((ss - 2)->contHistory.at(chIndex), bonus);
    if ((ss - 4)->playedMove)
      addToHistory((ss - 4)->contHistory.at(chIndex), bonus);
    if ((ss - 6)->playedMove)
      addToHistory((ss - 6)->contHistory.at(chIndex), bonus);
  }

  void Thread::updateHistories(Position& pos, int bonus, int malus, Move bestMove,
                       Move* quiets, int quietCount, int depth, SearchInfo* ss) {

    // Counter move
    if ((ss - 1)->playedMove) {
      Square prevSq = move_to((ss - 1)->playedMove);
      counterMoveHistory[pos.board[prevSq] * SQUARE_NB + prevSq] = bestMove;
    }

    // Killer move
    ss->killerMove = bestMove;

    // Credits to Ethereal
    // Don't prop up the best move if it was a quick low depth cutoff
    if (depth <= 3 && !quietCount)
      return;

    // Butterfly history
    addToHistory(mainHistory[pos.sideToMove][move_from_to(bestMove)], bonus);

    // Continuation history
    addToContHistory(pos, bonus, bestMove, ss);

    // Decrease score of other quiet moves
    for (int i = 0; i < quietCount; i++) {
      Move otherMove = quiets[i];
      addToContHistory(pos, -malus, otherMove, ss);
      addToHistory(mainHistory[pos.sideToMove][move_from_to(otherMove)], -malus);
    }
  }

  bool canUseScore(TT::Flag bound, Score score, Score operand) {
    return bound & (score >= operand ? TT::FLAG_LOWER : TT::FLAG_UPPER);
  }

  bool Thread::hasUpcomingRepetition(Position& pos, int ply) {

    const Bitboard occ = pos.pieces();
    const int maxDist = std::min(pos.halfMoveClock, keyStackHead);

    for (int i = 3; i <= maxDist; i += 2) {

      Key moveKey = pos.key ^ keyStack[keyStackHead - i];

      int hash = Cuckoo::h1(moveKey);

      // try the other slot
      if (Cuckoo::keys[hash] != moveKey)
        hash = Cuckoo::h2(moveKey);

      if (Cuckoo::keys[hash] != moveKey)
        continue; // neither slot matches

      Move   move = Cuckoo::moves[hash];
      Square from = move_from(move);
      Square to = move_to(move);

      // Check if the move is obstructed
      if ((BETWEEN_BB[from][to] ^ to) & occ)
        continue;

      // Repetition after root
      if (ply > i)
        return true;

      Piece pc = pos.board[ pos.board[from] ? from : to ];

      if (piece_color(pc) != pos.sideToMove)
        continue;

      // We want one more repetition before root
      for (int j = i+4; j <= maxDist; j += 2) {
        if (keyStack[keyStackHead - j] == keyStack[keyStackHead - i])
          return true;
      }
    }

    return false;
  }

  bool Thread::isRepetition(Position& pos, int ply) {

    const int maxDist = std::min(pos.halfMoveClock, keyStackHead);

    bool hitBeforeRoot = false;

    for (int i = 4; i <= maxDist; i += 2) {
      if (pos.key == keyStack[keyStackHead - i]) {
        if (ply >= i)
          return true;
        if (hitBeforeRoot)
          return true;
        hitBeforeRoot = true;
      }
    }

    return false;
  }

  Score Thread::qsearch(bool IsPV, Position& pos, Score alpha, Score beta, int depth, SearchInfo* ss) {

    // Detect upcoming draw
    if (alpha < SCORE_DRAW && hasUpcomingRepetition(pos, ply)) {
      alpha = SCORE_DRAW;
      if (alpha >= beta)
        return alpha;
    }

    // Detect draw
    if (isRepetition(pos, ply) || pos.is50mrDraw())
      return SCORE_DRAW;

    // Quit if we are close to reaching max ply
    if (ply >= MAX_PLY-4)
      return pos.checkers ? SCORE_DRAW : adjustEval(pos, doEvaluation(pos));

    // Probe TT
    const Key posTtKey = pos.key ^ ZOBRIST_50MR[pos.halfMoveClock];
    bool ttHit;
    TT::Entry* ttEntry = TT::probe(posTtKey, ttHit);
    TT::Flag ttBound = TT::NO_FLAG;
    Score ttScore = SCORE_NONE;
    Move ttMove = MOVE_NONE;
    Score ttStaticEval = SCORE_NONE;
    bool ttPV = false;

    if (ttHit) {
      ttBound = ttEntry->getBound();
      ttScore = ttEntry->getScore(ply);
      ttMove = ttEntry->getMove();
      ttStaticEval = ttEntry->getStaticEval();
      ttPV = ttEntry->wasPV();
    }

    // In non PV nodes, if tt bound allows it, return ttScore
    if ( !IsPV
      && ttScore != SCORE_NONE
      && canUseScore(ttBound, ttScore, beta))
        return ttScore;

    Move bestMove = MOVE_NONE;
    Score rawStaticEval;
    Score bestScore;
    Score futility;

    // Do the static evaluation

    if (pos.checkers) {
      // When in check avoid evaluating
      bestScore = -SCORE_INFINITE;
      futility = ss->staticEval = rawStaticEval = SCORE_NONE;
    }
    else {
      if (ttStaticEval != SCORE_NONE)
        rawStaticEval = ttStaticEval;
      else
        rawStaticEval = doEvaluation(pos);

      bestScore = ss->staticEval = adjustEval(pos, rawStaticEval);

      futility = bestScore + QsFpMargin;

      // When tt bound allows it, use ttScore as a better standing pat
      if (ttScore != SCORE_NONE && canUseScore(ttBound, ttScore, bestScore))
        bestScore = ttScore;

      if (bestScore >= beta) {
        if (! ttHit)
          ttEntry->store(posTtKey, TT::NO_FLAG, 0, MOVE_NONE, SCORE_NONE, rawStaticEval, false, ply);
        return (bestScore + beta) / 2;
      }
      if (bestScore > alpha)
        alpha = bestScore;
    }

    MovePicker movePicker(
      MovePicker::QSEARCH, pos,
      ttMove, MOVE_NONE, MOVE_NONE,
      mainHistory, captureHistory,
      0,
      ss);

    movePicker.genQuietChecks = (depth == 0);

    bool foundLegalMoves = false;

    // Visit moves
    Move move;

    while (move = movePicker.nextMove(false)) {

      TT::prefetch(pos.keyAfter(move));

      if (!pos.isLegal(move))
        continue;

      foundLegalMoves = true;

      bool isQuiet = pos.isQuiet(move);

      if (bestScore > SCORE_TB_LOSS_IN_MAX_PLY) {
        if (!isQuiet && !pos.checkers && futility <= alpha && !pos.seeGe(move, 1)) {
          bestScore = std::max(bestScore, futility);
          continue;
        }

        if (!pos.seeGe(move, QsSeeMargin))
          continue;
      }

      Position newPos = pos;
      playMove(newPos, move, ss);

      Score score = -qsearch(IsPV, newPos, -beta, -alpha, depth - 1, ss + 1);

      cancelMove();

      if (score > bestScore) {
        bestScore = score;

        if (bestScore > alpha) {
          bestMove = move;

          // Always true in NonPV nodes
          if (bestScore >= beta)
            break;

          alpha = bestScore;
        }
      }

      if (bestScore > SCORE_TB_LOSS_IN_MAX_PLY) {
        if (pos.checkers && isQuiet)
          break;
      }
    }

    if (pos.checkers && !foundLegalMoves)
      return ply - SCORE_MATE;

    if (bestScore >= beta && myabs(bestScore) < SCORE_TB_WIN_IN_MAX_PLY)
      bestScore = (bestScore + beta) / 2;

    ttEntry->store(posTtKey,
      bestScore >= beta ? TT::FLAG_LOWER : TT::FLAG_UPPER,
      0, bestMove, bestScore, rawStaticEval, ttPV, ply);

    return bestScore;
  }

  void updatePV(SearchInfo* ss, int ply, Move move) {

    ss->pvLength = (ss + 1)->pvLength;

    // set the move in the pv
    ss->pv[ply] = move;

    // copy all the moves that follow, from the child pv
    for (int i = ply + 1; i < (ss + 1)->pvLength; i++)
      ss->pv[i] = (ss + 1)->pv[i];
  }

  Score Thread::negamax(bool IsPV, Position& pos, Score alpha, Score beta, int depth, bool cutNode, SearchInfo* ss, const Move excludedMove) {

    const bool IsRoot = IsPV && ply == 0;

    // Check time
    ++maxTimeCounter;
    if ( this == Threads::mainThread()
      && (maxTimeCounter & 4095) == 0
      && elapsedTime() >= maxTime)
        Threads::stopSearch();

    if (Threads::isSearchStopped())
      return SCORE_DRAW;

    // Init node
    if (IsPV)
      ss->pvLength = ply;

    // Enter qsearch when depth is 0
    if (depth <= 0)
      return qsearch(IsPV, pos, alpha, beta, 0, ss);

    // Detect upcoming draw
    if (!IsRoot && alpha < SCORE_DRAW && hasUpcomingRepetition(pos, ply)) {
      alpha = SCORE_DRAW;
      if (alpha >= beta)
        return alpha;
    }

    // Detect draw
    if (!IsRoot && (isRepetition(pos, ply) || pos.is50mrDraw()))
      return SCORE_DRAW;

    // Quit if we are close to reaching max ply
    if (ply >= MAX_PLY - 4)
      return pos.checkers ? SCORE_DRAW : adjustEval(pos, doEvaluation(pos));

    // Mate distance pruning
    alpha = std::max(alpha, ply - SCORE_MATE);
    beta = std::min(beta, SCORE_MATE - ply - 1);
    if (alpha >= beta)
      return alpha;

    // Probe TT
    const Key posTtKey = pos.key ^ ZOBRIST_50MR[pos.halfMoveClock];
    bool ttHit;
    TT::Entry* ttEntry = TT::probe(posTtKey, ttHit);

    TT::Flag ttBound = TT::NO_FLAG;
    Score ttScore   = SCORE_NONE;
    Move ttMove     = MOVE_NONE;
    int ttDepth     = -1;
    Score ttStaticEval = SCORE_NONE;
    bool ttPV = IsPV;

    if (ttHit) {
      ttBound = ttEntry->getBound();
      ttScore = ttEntry->getScore(ply);
      ttMove = ttEntry->getMove();
      ttDepth = ttEntry->getDepth();
      ttStaticEval = ttEntry->getStaticEval();
      ttPV |= ttEntry->wasPV();
    }

    if (IsRoot)
      ttMove = rootMoves[pvIdx].move;

    const bool ttMoveNoisy = ttMove && !pos.isQuiet(ttMove);

    const Score probcutBeta = beta + ProbcutBetaMargin;

    Score eval;
    Move bestMove = MOVE_NONE;
    Score rawStaticEval = SCORE_NONE;
    Score bestScore = -SCORE_INFINITE;
    Score maxScore  =  SCORE_INFINITE;

    // In non PV nodes, if tt depth and bound allow it, return ttScore
    if ( !IsPV
      && !excludedMove
      && ttScore != SCORE_NONE
      && ttDepth >= depth
      && canUseScore(ttBound, ttScore, beta)
      && pos.halfMoveClock < 90) // The TT entry might trick us into thinking this is not a draw
        return ttScore;

    (ss + 1)->killerMove = MOVE_NONE;

    bool improving = false;

    // Do the static evaluation

    if (pos.checkers) {
      // When in check avoid evaluating and skip pruning
      ss->staticEval = eval = SCORE_NONE;
      goto moves_loop;
    }
    else if (excludedMove) {
      // We have already evaluated the position in the node which invoked this singular search
      updateAccumulator(pos, accumStack[accumStackHead]);
      rawStaticEval = eval = ss->staticEval;
    }
    else {
      if (ttStaticEval != SCORE_NONE) {
        rawStaticEval = ttStaticEval;
        if (IsPV)
          updateAccumulator(pos, accumStack[accumStackHead]);
      }
      else
        rawStaticEval = doEvaluation(pos);

      eval = ss->staticEval = adjustEval(pos, rawStaticEval);

      if (!ttHit) {
        // This (probably new) position has just been evaluated.
        // Immediately save the evaluation in TT, so other threads who reach this position
        // won't need to evaluate again
        // This is also helpful when we cutoff early and no other store will be performed
        ttEntry->store(posTtKey, TT::NO_FLAG, 0, MOVE_NONE, SCORE_NONE, rawStaticEval, ttPV, ply);
      }

      // When tt bound allows it, use ttScore as a better evaluation
      if (ttScore != SCORE_NONE && canUseScore(ttBound, ttScore, eval))
        eval = ttScore;
    }

    if (!(ss-1)->playedCap && (ss-1)->staticEval != SCORE_NONE) {
      int theirLoss = (ss-1)->staticEval + ss->staticEval - EvalHistA;
      int bonus = std::clamp(EvalHistB * theirLoss / 64, -EvalHistC, EvalHistC);
      addToHistory(mainHistory[~pos.sideToMove][move_from_to((ss-1)->playedMove)], bonus);
    }

    // Calculate whether the evaluation here is worse or better than it was 2 plies ago
    if ((ss - 2)->staticEval != SCORE_NONE)
      improving = ss->staticEval > (ss - 2)->staticEval;
    else if ((ss - 4)->staticEval != SCORE_NONE)
      improving = ss->staticEval > (ss - 4)->staticEval;

    // Razoring. When evaluation is far below alpha, we could probably only catch up with a capture,
    // thus do a qsearch. If the qsearch still can't hit alpha, cut off
    if ( !IsPV
      && alpha < 2000
      && eval < alpha - RazoringDepthMul * depth) {
      Score score = qsearch(IsPV, pos, alpha, beta, 0, ss);
      if (score <= alpha)
        return score;
    }

    // Reverse futility pruning. When evaluation is far above beta, assume that at least a move
    // will return a similarly high score, so cut off
    if ( !IsPV
      && depth <= RfpMaxDepth
      && eval < SCORE_TB_WIN_IN_MAX_PLY
      && eval - std::max(RfpDepthMul * (depth - improving), 20) >= beta)
      return (eval + beta) / 2;

    // Null move pruning. When our evaluation is above beta, we give the opponent
    // a free move, and if we are still better, cut off
    if ( !IsPV
      && !excludedMove
      && (ss - 1)->playedMove != MOVE_NONE
      && eval >= beta
      && ss->staticEval + NmpA * depth - NmpB >= beta
      && pos.hasNonPawns(pos.sideToMove)
      && beta > SCORE_TB_LOSS_IN_MAX_PLY) {

      TT::prefetch(pos.key ^ ZOBRIST_TEMPO);

      int R = std::min((eval - beta) / NmpEvalDiv, (int)NmpEvalDivMin) + depth / NmpDepthDiv + NmpBase + ttMoveNoisy;

      Position newPos = pos;
      playNullMove(newPos, ss);
      Score score = -negamax(false, newPos, -beta, -beta + 1, depth - R, !cutNode, ss + 1);
      cancelNullMove();

      if (score >= beta)
        return score < SCORE_TB_WIN_IN_MAX_PLY ? score : beta;
    }

    // IIR. Decrement the depth if we expect this search to have bad move ordering
    if ((IsPV || cutNode) && depth >= 2+2*cutNode && !ttMove)
      depth--;

    if (   !IsPV
        && depth >= 5
        && myabs(beta) < SCORE_TB_WIN_IN_MAX_PLY
        && !(ttDepth >= depth - 3 && ttScore < probcutBeta))
    {
      int pcSeeMargin = (probcutBeta - ss->staticEval) * 10 / 16;
      bool visitTTMove = ttMoveNoisy && pos.seeGe(ttMove, pcSeeMargin);

      MovePicker pcMovePicker(
        MovePicker::PROBCUT, pos,
        visitTTMove ? ttMove : MOVE_NONE, MOVE_NONE, MOVE_NONE,
        mainHistory, captureHistory,
        pcSeeMargin,
        ss);

      Move move;

      while (move = pcMovePicker.nextMove(false)) {

        TT::prefetch(pos.keyAfter(move));

        if (!pos.isLegal(move))
          continue;

        Position newPos = pos;
        playMove(newPos, move, ss);

        Score score = -qsearch(false, newPos, -probcutBeta, -probcutBeta + 1, 0, ss + 1);

        // Do a normal search if qsearch was positive
        if (score >= probcutBeta)
          score = -negamax(false, newPos, -probcutBeta, -probcutBeta + 1, depth - 4, !cutNode, ss + 1);

        cancelMove();

        if (Threads::isSearchStopped())
          return SCORE_DRAW;

        if (score >= probcutBeta) {
          ttEntry->store(posTtKey, TT::FLAG_LOWER, depth - 3, move, score, rawStaticEval, ttPV, ply);
          return score;
        }
      }
    }

  moves_loop:

    // Generate moves and score them

    bool skipQuiets = false;
    int seenMoves = 0;

    Move quiets[64];
    int quietCount = 0;
    Move captures[64];
    int captureCount = 0;

    Move counterMove = MOVE_NONE;
    if ((ss - 1)->playedMove) {
      Square prevSq = move_to((ss - 1)->playedMove);
      counterMove = counterMoveHistory[pos.board[prevSq] * SQUARE_NB + prevSq];
    }

    if (IsRoot)
      ss->killerMove = MOVE_NONE;

    MovePicker movePicker(
      MovePicker::PVS, pos,
      ttMove, ss->killerMove, counterMove,
      mainHistory, captureHistory,
      0,
      ss);

    // Visit moves

    Move move;

    while (move = movePicker.nextMove(skipQuiets)) {
      if (move == excludedMove)
        continue;

      TT::prefetch(pos.keyAfter(move));

      if (!pos.isLegal(move))
        continue;

      if (IsRoot && !visitRootMove(move))
        continue;

      seenMoves++;

      bool isQuiet = pos.isQuiet(move);

      int history = isQuiet ? getQuietHistory(pos, move, ss) : getCapHistory(pos, move);

      int oldNodesSearched = nodesSearched;

      if ( !IsRoot
        && bestScore > SCORE_TB_LOSS_IN_MAX_PLY
        && pos.hasNonPawns(pos.sideToMove))
      {
        int lmrRed = lmrTable[depth][seenMoves] + !improving - history / EarlyLmrHistoryDiv;
        int lmrDepth = std::max(0, depth - lmrRed);

        // SEE (Static Exchange Evalution) pruning
        int seeMargin = isQuiet ? PvsQuietSeeMargin * lmrDepth * lmrDepth :
                                  PvsCapSeeMargin * depth;
        if (!pos.seeGe(move, seeMargin))
          continue;

        if (isQuiet && history < HistPrDepthMul * depth)
            skipQuiets = true;

        // Late move pruning. At low depths, only visit a few quiet moves
        if (seenMoves >= (depth * depth + LmpBase) / (2 - improving))
          skipQuiets = true;

        // Futility pruning. If our evaluation is far below alpha,
        // only visit a few quiet moves
        if (   isQuiet
            && lmrDepth <= FpMaxDepth
            && !pos.checkers
            && ss->staticEval + FpBase + FpDepthMul * lmrDepth <= alpha) {
          skipQuiets = true;
          continue;
        }
      }

      int extension = 0;

      // Singular extension
      if ( !IsRoot
        && ply < 2 * rootDepth
        && depth >= 5
        && !excludedMove
        && move == ttMove
        && myabs(ttScore) < SCORE_TB_WIN_IN_MAX_PLY
        && ttBound & TT::FLAG_LOWER
        && ttDepth >= depth - 3)
      {
        Score singularBeta = ttScore - (depth * SBetaMargin) / 64;

        Score seScore = negamax(false, pos, singularBeta - 1, singularBeta, (depth - 1) / 2, cutNode, ss, move);

        if (seScore < singularBeta) {
          // Extend even more if s. value is smaller than s. beta by some margin
          if (   !IsPV
              && seScore < singularBeta - DoubleExtMargin)
          {
            extension = 2 + (isQuiet && seScore < singularBeta - TripleExtMargin);
          } else {
            extension = 1;
          }
        }
        else if (singularBeta >= beta) // Multicut
          return singularBeta;
        else if (ttScore >= beta) // Negative extensions
          extension = -2 + IsPV;
        else if (cutNode)
          extension = -2;
      }

      Position newPos = pos;
      playMove(newPos, move, ss);

      int newDepth = depth + extension - 1;

      Score score;

      // Late move reductions

      if (depth >= 2 && seenMoves > 1 + 2 * IsRoot) {

        int R = lmrTable[depth][seenMoves];

        R -= history / (isQuiet ? LmrQuietHistoryDiv : LmrCapHistoryDiv);

        R -= (newPos.checkers != 0ULL);

        R -= (ttDepth >= depth);

        R -= ttPV + IsPV;

        R += ttMoveNoisy;

        R += !improving;

        R += 2 * cutNode;

        // Clamp to avoid a qsearch or an extension in the child search
        int reducedDepth = std::clamp(newDepth - R, 1, newDepth + 1);

        score = -negamax(false, newPos, -alpha - 1, -alpha, reducedDepth, true, ss + 1);

        if (score > alpha && reducedDepth < newDepth) {
          newDepth += (score > bestScore + ZwsDeeperMargin);
          newDepth -= (score < bestScore + newDepth        && !IsRoot);

          if (reducedDepth < newDepth)
            score = -negamax(false, newPos, -alpha - 1, -alpha, newDepth, !cutNode, ss + 1);

          int bonus = score <= alpha ? -statMalus(newDepth) : score >= beta ? statBonus(newDepth) : 0;
          addToContHistory(pos, bonus, move, ss);
        }
      }
      else if (!IsPV || seenMoves > 1)
        score = -negamax(false, newPos, -alpha - 1, -alpha, newDepth, !cutNode, ss + 1);

      if (IsPV && (seenMoves == 1 || score > alpha))
        score = -negamax(true, newPos, -beta, -alpha, newDepth, false, ss + 1);

      cancelMove();

      if (Threads::isSearchStopped())
        return SCORE_DRAW;

      if (IsRoot) {
        RootMove& rm = rootMoves[rootMoves.indexOf(move)];
        rm.nodes += nodesSearched - oldNodesSearched;

        if (seenMoves == 1 || score > alpha) {
          rm.score = score;

          rm.pvLength = (ss+1)->pvLength;
          rm.pv[0] = move;
          for (int i = 1; i < (ss+1)->pvLength; i++)
            rm.pv[i] = (ss+1)->pv[i];
        }
        else // this move gave an upper bound, so we don't know how to sort it
          rm.score = - SCORE_INFINITE;
      }

      if (score > bestScore) {
        bestScore = score;

        if (bestScore > alpha) {
          bestMove = move;

          if (IsPV && !IsRoot)
            updatePV(ss, ply, bestMove);

          // Always true in NonPV nodes
          if (bestScore >= beta)
            break;

          alpha = bestScore;
        }
      }

      // Register the move to decrease its history later. Unless it raised alpha
      if (move != bestMove) {
        if (isQuiet) {
          if (quietCount < 64)
            quiets[quietCount++] = move;
        }
        else {
          if (captureCount < 64)
            captures[captureCount++] = move;
        }
      }
    }

    if (!seenMoves) {
      if (excludedMove)
        return alpha;

      return pos.checkers ? ply - SCORE_MATE : SCORE_DRAW;
    }

    // Only in pv nodes we could probe TB and not cut off immediately
    if (IsPV)
      bestScore = std::min(bestScore, maxScore);

    // Update histories
    if (bestScore >= beta)
    {
      int bonus = statBonus(depth + (bestScore > beta + StatBonusBoostAt));
      int malus = statMalus(depth + (bestScore > beta + StatBonusBoostAt));

      if (pos.isQuiet(bestMove))
      {
        updateHistories(pos, bonus, malus, bestMove, quiets, quietCount, depth, ss);
      }
      else {
        PieceType captured = piece_type(pos.board[move_to(bestMove)]);
        addToHistory(captureHistory[pieceTo(pos, bestMove)][captured], bonus);
      }

      for (int i = 0; i < captureCount; i++) {
        Move otherMove = captures[i];
        PieceType captured = piece_type(pos.board[move_to(otherMove)]);
        addToHistory(captureHistory[pieceTo(pos, otherMove)][captured], -malus);
      }
    }

    TT::Flag resultBound;
    if (bestScore >= beta)
      resultBound = TT::FLAG_LOWER;
    else
      resultBound = (IsPV && bestMove) ? TT::FLAG_EXACT : TT::FLAG_UPPER;

    // update corrhist
    const bool bestMoveCap = pos.board[move_to(bestMove)] != NO_PIECE;
    if (   !pos.checkers
        && !(bestMove && bestMoveCap)
        && canUseScore(resultBound, bestScore, ss->staticEval))
    {
      int bonus = std::clamp((bestScore - ss->staticEval) * depth / 8,
                             -CORRHIST_LIMIT / 4, CORRHIST_LIMIT / 4);
                  
      addToCorrhist(pawnCorrhist[pos.sideToMove][getCorrHistIndex(pos.pawnKey)], bonus);
      addToCorrhist(wNonPawnCorrhist[pos.sideToMove][getCorrHistIndex(pos.nonPawnKey[WHITE])], bonus);
      addToCorrhist(bNonPawnCorrhist[pos.sideToMove][getCorrHistIndex(pos.nonPawnKey[BLACK])], bonus);
    }

    // Store to TT
    if (!excludedMove && !(IsRoot && pvIdx > 0))
      ttEntry->store(posTtKey, resultBound, depth, bestMove, bestScore, rawStaticEval, ttPV, ply);

    return bestScore;
  }

#ifndef SHARED
  std::string getPvString(RootMove& rm) {

    std::ostringstream output;

    output << UCI::moveToString(rm.move);

    for (int i = 1; i < rm.pvLength; i++) {
      Move move = rm.pv[i];
      if (!move)
        break;

      output << ' ' << UCI::moveToString(move);
    }

    return output.str();
  }
#endif
  
  DEFINE_PARAM_B(tm0, 192, 50, 200);
  DEFINE_PARAM_B(tm1, 61,  20, 100);

  DEFINE_PARAM_B(tm2, 169, 50, 200);
  DEFINE_PARAM_B(tm3, 7,   0, 30);

  DEFINE_PARAM_B(tm4, 84,   0, 150);
  DEFINE_PARAM_B(tm5, 10,   0, 20);
  DEFINE_PARAM_B(tm6, 26,   0, 50);

  DEFINE_PARAM_B(lol0, 81, 0, 150);
  DEFINE_PARAM_B(lol1, 146,   75,  225);

  void Thread::startSearch() {

    const Settings& settings = Threads::getSearchSettings();

    Position rootPos = settings.position;

    accumStackHead = 0;
    for (Color side = WHITE; side <= BLACK; ++side) {
      accumStack[0].refresh(rootPos, side);
      accumStack[0].kings[side] = rootPos.kingSquare(side);
    }

    for (int i = 0; i < 2; i++)
      finny[i].reset();

    keyStackHead = 0;
    for (int i = 0; i < settings.prevPositions.count; i++)
      keyStack[keyStackHead++] = settings.prevPositions[i];

    maxTime = 999999999999LL;

    if (settings.standardTimeLimit()) {
      int64_t stdMaxTime;
      TimeMan::calcOptimumTime(settings, rootPos.sideToMove, &optimumTime, &stdMaxTime);
      maxTime = std::min(maxTime, stdMaxTime);
    }
    if (settings.movetime)
      maxTime = std::min(maxTime, settings.movetime - int64_t(10));

    ply = 0;
    maxTimeCounter = 0;

    // Setup search stack

    SearchInfo* ss = &searchStack[SsOffset];

    for (int i = 0; i < MAX_PLY + SsOffset; i++) {
      searchStack[i].staticEval = SCORE_NONE;

      searchStack[i].pvLength = 0;

      searchStack[i].killerMove = MOVE_NONE;
      searchStack[i].playedMove = MOVE_NONE;
      searchStack[i].playedCap = false;

      searchStack[i].contHistory = {
        .table = contHistory[false],
        .hash = 0
      };
    }

    bool naturalExit = true;

    // TM variables
    Move idPrevMove = MOVE_NONE;
    Score idPrevScore = SCORE_NONE;
    int searchStability = 0;

    // Setup root moves
    rootMoves = RootMoveList();
    {
      MoveList pseudoRootMoves;
      getStageMoves(rootPos, ADD_ALL_MOVES, &pseudoRootMoves);

      for (int i = 0; i < pseudoRootMoves.size(); i++) {
        Move move = pseudoRootMoves[i].move;
        if (rootPos.isLegal(move))
          rootMoves.add(move);
      }
    }

    // Search starting. Zero out the nodes of each root move
    for (int i = 0; i < rootMoves.size(); i++)
      rootMoves[i].nodes = 0;

    const int multiPV = std::min(1, rootMoves.size());

    for (rootDepth = 1; rootDepth <= settings.depth; rootDepth++) {

      // Only one legal move? For analysis purposes search, but with a limited depth
      if (rootDepth > 10 && rootMoves.size() == 1)
        break;

      for (pvIdx = 0; pvIdx < multiPV; pvIdx++) {
        int window = AspWindowStartDelta;
        Score alpha = -SCORE_INFINITE;
        Score beta  = SCORE_INFINITE;
        int failHighCount = 0;

        if (rootDepth >= AspWindowStartDepth) {
          alpha = std::max(-SCORE_INFINITE, rootMoves[pvIdx].score - window);
          beta  = std::min( SCORE_INFINITE, rootMoves[pvIdx].score + window);
        }

        while (true) {

          int adjustedDepth = std::max(1, rootDepth - failHighCount);

          Score score = negamax(true, rootPos, alpha, beta, adjustedDepth, false, ss);

          // The score of any root move is updated only if search wasn't yet stopped at the moment of updating.
          // This means that the root moves' score is usable at any time
          sortRootMoves(pvIdx);

          if (Threads::isSearchStopped()) {
            naturalExit = false;
            goto bestMoveDecided;
          }

          if (score <= alpha) {
            beta = (alpha + beta) / 2;
            alpha = std::max(-SCORE_INFINITE, score - window);

            failHighCount = 0;
          }
          else if (score >= beta) {
            beta = std::min(SCORE_INFINITE, score + window);

            if (score < 2000)
              failHighCount++;
          }
          else
            break;

          if (settings.nodes && Threads::totalNodes() >= settings.nodes) {
            naturalExit = false;
            goto bestMoveDecided;
          }

          window += window / 3;
        }

        sortRootMoves(0);
      }

      completeDepth = rootDepth;

      if (settings.nodes && Threads::totalNodes() >= settings.nodes) {
        naturalExit = false;
        goto bestMoveDecided;
      }

      if (this != Threads::mainThread())
        continue;

      const int64_t elapsed = elapsedTime();

#ifndef SHARED
      if (std::string(UCI::Options["Minimal"]) != "true") {
        for (int i = 0; i < multiPV; i++)
          printInfo(completeDepth, i+1, rootMoves[i].score, getPvString(rootMoves[i]));
      }
#endif

      if (elapsedTime() >= maxTime)
        goto bestMoveDecided;

      const Move bestMove = rootMoves[0].move;
      const Score score = rootMoves[0].score;

      if (bestMove == idPrevMove)
        searchStability = std::min(searchStability + 1, 8);
      else
        searchStability = 0;

      if (settings.standardTimeLimit() && rootDepth >= 4) {
        int bmNodes = rootMoves[rootMoves.indexOf(bestMove)].nodes;
        double notBestNodes = 1.0 - (bmNodes / double(nodesSearched));
        double nodesFactor     = (tm1/100.0) + notBestNodes * (tm0/100.0);

        double stabilityFactor = (tm2/100.0) - searchStability * (tm3/100.0);

        double scoreLoss =   (tm4/100.0)
                           + (tm5/1000.0) * (idPrevScore     - score)
                           + (tm6/1000.0) * (searchPrevScore - score);

        double scoreFactor = std::clamp(scoreLoss, lol0 / 100.0, lol1 / 100.0);

        if (elapsed > stabilityFactor * nodesFactor * scoreFactor * optimumTime)
          goto bestMoveDecided;
      }

      idPrevMove = bestMove;
      idPrevScore = score;
    }

  bestMoveDecided:

    // NOTE: When implementing best thread selection, don't mess up with tablebases dtz stuff

    if (this != Threads::mainThread())
      return;

    Threads::stopSearch();

    // Threads::waitForSearch(false);

    Search::Thread* bestThread = this;

#ifndef SHARED
    if (!naturalExit || bestThread != this || std::string(UCI::Options["Minimal"]) == "true")
        for (int i = 0; i < multiPV; i++)
          printInfo(bestThread->completeDepth, i+1, bestThread->rootMoves[i].score, getPvString(bestThread->rootMoves[i]));
#endif
    
    searchPrevScore = bestThread->rootMoves[0].score;
    
    printBestMove(bestThread->rootMoves[0].move);
  }
}
