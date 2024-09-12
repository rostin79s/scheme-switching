#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "../frontend/dag.hpp"

void generateMainFunction(llvm::LLVMContext &Context, llvm::Module &module);

#endif