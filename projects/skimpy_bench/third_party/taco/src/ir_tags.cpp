#include "taco/ir_tags.h"

namespace taco {
const char *ParallelUnit_NAMES[] = {"NotParallel", "DefaultUnit", "GPUBlock", "GPUWarp", "GPUThread", "CPUThread", "CPUVector", "CPUThreadGroupReduction", "GPUBlockReduction", "GPUWarpReduction"};
const char *OutputRaceStrategy_NAMES[] = {"IgnoreRaces", "NoRaces", "Atomics", "Temporary", "ParallelReduction"};
const char *BoundType_NAMES[] = {"MinExact", "MinConstraint", "MaxExact", "MaxConstraint"};
}
