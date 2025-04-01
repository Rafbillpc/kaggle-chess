#pragma once

#include "history.h"
#include "search.h"

namespace Threads {
  extern Search::Thread searchThread;

  Search::Thread* mainThread();

  bool isSearchStopped();

  uint64_t totalNodes();

  uint64_t totalTbHits();

  // void waitForSearch(bool waitMain = true);

  void startSearch(Search::Settings& settings);

  Search::Settings& getSearchSettings();

  void stopSearch();

  void setThreadCount();
}
