#ifndef FRONTEND_HPP
#define FRONTEND_HPP

#include "dag.hpp"

std::unordered_map<std::string, std::string> getCiphertextArguments(llvm::Function &F);

void naming(DAG* dag);

std::string demangle(const std::string &mangledName);

DAG* buildDAGFromInstructions(llvm::Function &F);

#endif