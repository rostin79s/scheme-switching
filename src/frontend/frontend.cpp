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
#include "mlir/IR/BuiltinOps.h"


#include <llvm/Demangle/Demangle.h>

#include <unordered_map>
#include <string>
#include <iostream>

#include "dag.hpp"
#include "frontend.hpp"

using namespace mlir;

// Utility function to get operand names and types from MLIR
std::unordered_map<std::string, std::string> getCiphertextArguments(mlir::func::FuncOp func) {
    std::unordered_map<std::string, std::string> ciphertextArgs;

    // Get function type
    auto functionType = func.getFunctionType();
    
    // Iterate over function arguments
    for (unsigned index = 0; index < functionType.getNumInputs(); ++index) {
        auto argType = functionType.getInput(index);
        std::string argName = "arg" + std::to_string(index); // Generate default argument names
        std::string argTypeStr;
        llvm::raw_string_ostream rso(argTypeStr);
        argType.print(rso); // Print the type of the argument
        ciphertextArgs[argName] = rso.str();
    }

    return ciphertextArgs;
}

void naming(DAG* dag) {

    dag->returnType = "FHE" + dag->returnType;

    // New map for function inputs
    std::unordered_map<std::string, std::string> updatedFunctionInputs;

    // Replace % with _tmp in function inputs and add "FHE" to the type
    for (const auto &entry : dag->functionInputs) {

        // Add "FHE" to the start of the type
        std::string updatedType = "FHE" + entry.second;

        // Store the updated name and type in the new map
        updatedFunctionInputs[entry.first] = updatedType;
    }

    // Replace the old map with the updated one
    dag->functionInputs = std::move(updatedFunctionInputs);

    // Replace % with _tmp in DAG nodes
    for (auto &node : dag->nodes) {
        // Update result name
        std::string newResult = node->result;
        size_t pos = newResult.find('%');
        if (pos != std::string::npos) {
            newResult.replace(pos, 1, "var");
        }
        node->result = newResult;

        // Update operand names
        for (auto &operand : node->operands) {
            std::string newOperand = operand;
            pos = newOperand.find('%');
            if (pos != std::string::npos) {
                if (newOperand[1] == 'a'){
                    newOperand.replace(pos, 1, "");
                }
                else{
                    newOperand.replace(pos, 1, "var");
                }
                
            }
            operand = newOperand;
        }
    }
}

// Utility function to demangle names, if necessary
std::string demangle(const std::string &mangledName) {
    char *demangledName = llvm::itaniumDemangle(mangledName.c_str());
    std::string result(demangledName);
    free(demangledName);
    size_t pos = result.find('(');

    if (pos != std::string::npos) {
        result = result.substr(0, pos);
    }

    std::cout << result << std::endl;
    return result;
}

// Function to process each MLIR operation and build the DAG
DAG* buildDAGFromInstructions(mlir::func::FuncOp func) {
    DAG *dag = new DAG();

    // Set function inputs and return type
    dag->functionInputs = getCiphertextArguments(func);
    dag->name = demangle(func.getName().str());

    

    // Set the return type
    auto returnType = func.getFunctionType().getResult(0);
    std::string rettype;
    llvm::raw_string_ostream rso(rettype);
    returnType.print(rso);
    dag->returnType = rso.str();

    // Create a MLIR builder for traversing operations

    // Iterate over blocks and operations in the function
    int id = 1;
    func->walk([&](mlir::Operation *op) {
            
        std::string fullName = op->getName().getStringRef().str();
        size_t dotPos = fullName.find('.');
        std::string operationStr = fullName.substr(dotPos + 1);
        std::vector<std::string> operandStrs;
        std::string operandType;
        std::string resultStr;

        // if (operationStr == "return") {
        //     return mlir::WalkResult::interrupt();
        // }


        // Extract result name and type
        if (op->getNumResults() > 0) {
            auto result = op->getResult(0);
            llvm::raw_string_ostream rso(resultStr);  
            mlir::OpPrintingFlags flags;
            result.printAsOperand(rso, flags);
            // resultStr = "var" + std::to_string(result.getResultNumber()); // Generate a result name
            
            // Extract the type of the result
            auto type = result.getType();
            llvm::raw_string_ostream typeStream(operandType);
            type.print(typeStream);
            operandType = typeStream.str();
        }

        // Extract operand names and types
        for (auto operand : op->getOperands()) {
            std::string temp;
            llvm::raw_string_ostream rso(temp);  
            mlir::OpPrintingFlags flags;
            operand.printAsOperand(rso, flags);
            if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                operandStrs.push_back(temp);
            } else if (operand.getDefiningOp()) {
                operandStrs.push_back(temp);
            } else {
                operandStrs.push_back("unknown");
            }
        }

        // Create a node for the operation
        DAGNode *node = dag->addNode(op, resultStr, operationStr, operandStrs, operandType, id);

        // Create edges based on operand dependencies
        for (auto operand : op->getOperands()) {
            if (auto *opInst = operand.getDefiningOp()) {
                if (dag->nodeMap.find(opInst) != dag->nodeMap.end()) {
                    DAGNode *opNode = dag->nodeMap[opInst];
                    dag->addEdge(opNode, node);
                }
            }
    
        }

        id++;

        return mlir::WalkResult::advance();
    });

    naming(dag);

    return dag;
}
