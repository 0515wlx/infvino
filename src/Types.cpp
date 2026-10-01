// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/Types.hpp"

#include <algorithm>
#include <cctype>

namespace infvino
{

const char * toString(Task task)
{
  switch (task)
  {
    case Task::Detect:   return "detect";
    case Task::Seg:      return "segment";
    case Task::Pose:     return "pose";
    case Task::Obb:      return "obb";
    case Task::Classify: return "classify";
    default:             return "unknown";
  }
}

Task taskFromString(const std::string & s)
{
  std::string t;
  t.reserve(s.size());
  std::transform(
    s.begin(), s.end(), std::back_inserter(t),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (t == "detect" || t == "detection") return Task::Detect;
  if (t == "segment" || t == "seg") return Task::Seg;
  if (t == "pose") return Task::Pose;
  if (t == "obb") return Task::Obb;
  if (t == "classify" || t == "cls") return Task::Classify;
  return Task::Unknown;
}

Task taskFromNumClasses(int /*nc*/)
{
  return Task::Unknown;
}

} // namespace infvino
