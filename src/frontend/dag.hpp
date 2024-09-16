#ifndef DAG_HPP
#define DAG_HPP

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/DLTI/DLTI.h"

#include <llvm/IR/Instruction.h>
#include <llvm/Support/raw_ostream.h>
#include <unordered_map>
#include <vector>
#include <iostream>

class DAGNode {
public:
    mlir::Operation *inst;
    std::string result;
    std::string operation;
    std::vector<std::string> operands;
    std::string operandType;
    std::vector<DAGNode*> dependencies; // List of nodes that this node depends on

    DAGNode(mlir::Operation *inst, const std::string &res, const std::string &op, 
            const std::vector<std::string> &ops, const std::string &type);

    void addDependency(DAGNode *node);

    void print() const;
};

// Class to represent the Directed Acyclic Graph (DAG)
class DAG {
public:
    std::unordered_map<mlir::Operation*, DAGNode*> nodeMap;
    std::vector<DAGNode*> nodes;
    std::unordered_map<std::string, std::string> functionInputs;
    std::string name;
    std::string returnType;

    DAGNode* addNode(mlir::Operation *inst, const std::string &res, const std::string &op,
                     const std::vector<std::string> &ops, const std::string &type);

    void addEdge(DAGNode *from, DAGNode *to);
    void print() const;
    void convert();


    DAG* optimize();
    void sort();
    void generateBackend(mlir::MLIRContext &context);
};

#endif