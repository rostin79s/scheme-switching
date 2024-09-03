#include <llvm/IR/Module.h>

#include "dag.hpp"


std::unordered_map<std::string, std::string> getCiphertextArguments(llvm::Function &F) {

    std::unordered_map<std::string, std::string> ciphertextArgs;

    for (auto &Arg : F.args()) {
        llvm::Value *result = &Arg;
        std::string resultStr;
        llvm::raw_string_ostream rso(resultStr);
        result->printAsOperand(rso, false);

        std::string paramType;
        llvm::raw_string_ostream rsoType(paramType);
        Arg.getType()->print(rsoType);

        ciphertextArgs[resultStr] = rsoType.str();

    }

    return ciphertextArgs;
}

void naming(DAG* dag) {

    dag->returnType = "FHE" + dag->returnType;

    // New map for function inputs
    std::unordered_map<std::string, std::string> updatedFunctionInputs;

    // Replace % with _tmp in function inputs and add "FHE" to the type
    for (const auto &entry : dag->functionInputs) {
        std::string oldName = entry.first;
        std::string newName = oldName;
        size_t pos = newName.find('%');
        if (pos != std::string::npos) {
            newName.replace(pos, 1, "_tmp");
        }

        // Add "FHE" to the start of the type
        std::string updatedType = "FHE" + entry.second;

        // Store the updated name and type in the new map
        updatedFunctionInputs[newName] = updatedType;
    }

    // Replace the old map with the updated one
    dag->functionInputs = std::move(updatedFunctionInputs);

    // Replace % with _tmp in DAG nodes
    for (auto &node : dag->nodes) {
        // Update result name
        std::string newResult = node->result;
        size_t pos = newResult.find('%');
        if (pos != std::string::npos) {
            newResult.replace(pos, 1, "_tmp");
        }
        node->result = newResult;

        // Update operand names
        for (auto &operand : node->operands) {
            std::string newOperand = operand;
            pos = newOperand.find('%');
            if (pos != std::string::npos) {
                newOperand.replace(pos, 1, "_tmp");
            }
            operand = newOperand;
        }
    }
}


// Function to process each instruction and build the DAG
DAG* buildDAGFromInstructions(llvm::Function &F) {

    DAG *dag = new DAG();

    dag->functionInputs = getCiphertextArguments(F);
    dag->name = F.getName().str();


    // added return type
    auto temp = F.getFunctionType()->getReturnType();
    std::string rettype;
    llvm::raw_string_ostream rso(rettype);
    temp->print(rso);
    dag->returnType = rso.str();

    // Iterate over instructions and build nodes and edges
    for (llvm::BasicBlock &BB : F) {
        for (llvm::Instruction &I : BB) {   

            // Extract result, operation, operands, and operand types
            llvm::Value *result = &I; // The result is the instruction itself
            std::string resultStr;
            llvm::raw_string_ostream rso(resultStr);
            result->printAsOperand(rso, false);

            std::string operationStr = I.getOpcodeName();
            std::vector<std::string> operandStrs;
            std::string operandType;

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
                operandType = rstoType.str();
            }

            // Create a node for the instruction
            DAGNode *node = dag->addNode(&I, resultStr, operationStr, operandStrs, operandType);

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

    naming(dag);

    return dag;
}