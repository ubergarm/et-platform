// clang-format off

#ifndef _ENTRY_POINT_H_
#define _ENTRY_POINT_H_
// Forward declaration of KernelArguments, user implementations will define
class KernelArguments;
extern "C" int entryPoint(KernelArguments * layer_dyn_info);

#endif 
