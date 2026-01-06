#include "sliding_window.h"

void WindowManager::pushKeyframe(const KFState& kf){
  win_.kfs.push_back(kf);
  if (win_.kfs.size() > max_kf_) {
    win_.kfs.pop_front();
  }
}
