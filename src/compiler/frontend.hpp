#ifndef FRONTEND_HPP
#define FRONTEND_HPP

#include "dag.hpp"

void printFunctionArguments(llvm::Function &F);

DAG* buildDAGFromInstructions(llvm::Function &F);

#endif