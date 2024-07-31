#include <llvm/IR/Module.h>

#include "dag.hpp"


void printFunctionArguments(llvm::Function &F) {
    llvm::errs() << "Function: " << F.getName() << "\n";

    for (auto &Arg : F.args()) {
        // Print the name of the parameter
        llvm::Value *result = &Arg; // The result is the instruction itself
        std::string resultStr;
        llvm::raw_string_ostream rso(resultStr);
        result->printAsOperand(rso, false);

        // Print the type of the parameter
        std::string paramType;
        llvm::raw_string_ostream rsoType(paramType);
        Arg.getType()->print(rsoType);

        // Output parameter information
        llvm::errs() << "Parameter: " << resultStr << ", Type: " << rsoType.str() << "\n";
    }
}


// Function to process each instruction and build the DAG
DAG* buildDAGFromInstructions(llvm::Function &F) {

    DAG *dag = new DAG();

    // Iterate over instructions and build nodes and edges
    for (llvm::BasicBlock &BB : F) {
        for (llvm::Instruction &I : BB) {
            // if (llvm::isa<llvm::AllocaInst>(&I)) {
            //     continue;
            // }

            // Extract result, operation, operands, and operand types
            llvm::Value *result = &I; // The result is the instruction itself
            std::string resultStr;
            llvm::raw_string_ostream rso(resultStr);
            result->printAsOperand(rso, false);

            std::string operationStr = I.getOpcodeName();
            std::vector<std::string> operandStrs;
            std::vector<std::string> operandTypes;

            for (unsigned int opIdx = 0; opIdx < I.getNumOperands(); ++opIdx) {
                llvm::Value *operand = I.getOperand(opIdx);

                // Convert operand to string
                std::string operandStr;
                llvm::raw_string_ostream rsoOperand(operandStr);
                operand->printAsOperand(rsoOperand, false);
                operandStrs.push_back(rsoOperand.str());

                // Convert operand type to string
                llvm::Type *type = operand->getType();
                std::string typeStr;
                llvm::raw_string_ostream rstoType(typeStr);
                type->print(rstoType);
                operandTypes.push_back(rstoType.str());
            }

            // Create a node for the instruction
            DAGNode *node = dag->addNode(&I, resultStr, operationStr, operandStrs, operandTypes);

            // Create edges based on operand dependencies
            for (unsigned int opIdx = 0; opIdx < I.getNumOperands(); ++opIdx) {
                llvm::Value *operand = I.getOperand(opIdx);
                if (llvm::Instruction *opInst = llvm::dyn_cast<llvm::Instruction>(operand)) {
                    if (dag->nodeMap.find(opInst) != dag->nodeMap.end()) {
                        DAGNode *opNode = dag->nodeMap[opInst];
                        dag->addEdge(opNode, node);
                    }
                }
            }
        }
    }

    return dag;
}