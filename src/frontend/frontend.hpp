#ifndef FRONTEND_HPP
#define FRONTEND_HPP

#include "dag.hpp"

using namespace mlir;
using namespace llvm;

std::unordered_map<std::string, std::string> getCiphertextArguments(mlir::func::FuncOp func);

void naming(DAG* dag);

void printOperation(mlir::Operation *op);

std::string demangle(const std::string &mangledName);

DAG* buildDAGFromInstructions(mlir::func::FuncOp func);

#endif