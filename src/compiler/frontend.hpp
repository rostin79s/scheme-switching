#ifndef FRONTEND_HPP
#define FRONTEND_HPP

#include "dag.hpp"

std::unordered_map<std::string, std::string> getCiphertextArguments(llvm::Function &F);

DAG* buildDAGFromInstructions(llvm::Function &F);

#endif