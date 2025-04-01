#include "threads.h"
#include "nnue.h"
#include "search.h"

namespace Threads {

  Search::Settings searchSettings;
  Search::Thread searchThread;
  bool searchStopped;

  Search::Thread* mainThread() {
    return &searchThread;
  }

  bool isSearchStopped() {
    return searchStopped;
  }

  uint64_t totalNodes() {
    uint64_t result = 0;
    result += searchThread.nodesSearched;
    return result;
  }

  uint64_t totalTbHits() {
    uint64_t result = 0;
    result += searchThread.tbHits;
    return result;
  }

  // void waitForSearch(bool waitMain) {
  //   for (int i = !waitMain; i < searchThreads.size(); i++) {
  //     Search::Thread* st = searchThreads[i];
  //     std::unique_lock lock(st->mutex);
  //     st->cv.wait(lock, [&] { return !st->searching; });
  //   }
  // }

  // void startSearchSingle(Search::Thread* st) {
  //   st->mutex.lock();
  //   st->searching = true;
  //   st->mutex.unlock();
  //   st->cv.notify_all();
  // }

  void startSearch(Search::Settings& settings) {
    searchSettings = settings;
    searchStopped = false;
    searchThread.nodesSearched = 0;
    searchThread.tbHits = 0;
    searchThread.completeDepth = 0;
    
    searchThread.startSearch();
    searchThread.searching = false;
  }

  Search::Settings& getSearchSettings() {
    return searchSettings;
  }

  void stopSearch() {
    searchStopped = true;
  }

  // std::atomic<int> startedThreadsCount;

  // void threadEntry(int index) {
  //   searchThreads[index] = new Search::Thread();
  //   startedThreadsCount++;
  //   searchThreads[index]->idleLoop();
  // }

  void setThreadCount() {
    searchThread.resetHistories();
  }
}
