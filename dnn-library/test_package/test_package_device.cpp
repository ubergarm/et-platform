#include <dnn_lib/LibTensor.h>

int main() {
  // force using symbols from header (99% of the times this will not be executed because this is a RISC-V test)
  auto result = dnn_lib::isQuantizedElemKind(dnn_lib::ElemKind::FloatTy);
  return result;
}