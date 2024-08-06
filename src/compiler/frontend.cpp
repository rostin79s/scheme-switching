#include <llvm/IR/Module.h>

#include "dag.hpp"


std::unordered_map<std::string, std::string> getCiphertextArguments(llvm::Function &F) {

    llvm::errs() << "Function: " << F.getName() << "\n";

    std::unordered_map<std::string, std::string> ciphertextArgs;

    for (auto &Arg : F.args()) {
        llvm::Value *result = &Arg;
        std::string resultStr;
        llvm::raw_string_ostream rso(resultStr);
        result->printAsOperand(rso, false);

        std::string paramType;
        llvm::raw_string_ostream rsoType(paramType);
        Arg.getType()->print(rsoType);

        // Here, we'll assume that we identify a ciphertext by its type or name.
        // For demonstration, let's assume that any type containing "Ciphertext" is a ciphertext.
        if (rsoType.str().find("Ciphertext") != std::string::npos) {
            ciphertextArgs[resultStr] = rsoType.str();
        }

        llvm::errs() << "Parameter: " << resultStr << ", Type: " << rsoType.str() << "\n";
    }

    return ciphertextArgs;
}


DAG* buildDAGFromInstructions(llvm::Function &F) {
    DAG *dag = new DAG();

    // Node map to store instructions to DAG nodes mapping
    std::unordered_map<llvm::Instruction*, DAGNode*> nodeMap;

    // First pass: Create all nodes
    for (llvm::BasicBlock &BB : F) {
        for (llvm::Instruction &I : BB) {
            llvm::Value *result = &I;
            std::string resultStr;
            llvm::raw_string_ostream rso(resultStr);
            result->printAsOperand(rso, false);

            std::string operationStr = I.getOpcodeName();
            std::vector<std::string> operandStrs;
            std::string operandType;

            for (unsigned int opIdx = 0; opIdx < I.getNumOperands(); ++opIdx) {
                llvm::Value *operand = I.getOperand(opIdx);

                std::string operandStr;
                llvm::raw_string_ostream rsoOperand(operandStr);
                operand->printAsOperand(rsoOperand, false);
                operandStrs.push_back(rsoOperand.str());

                llvm::Type *type = operand->getType();
                std::string typeStr;
                llvm::raw_string_ostream rstoType(typeStr);
                type->print(rstoType);
                operandType = rstoType.str();
            }

            // Add the node only if it doesn't already exist
            DAGNode* node;
            if (nodeMap.find(&I) == nodeMap.end()) {
                node = dag->addNode(resultStr, operationStr, operandStrs, operandType);
                nodeMap[&I] = node;
            }

            for (unsigned int opIdx = 0; opIdx < I.getNumOperands(); ++opIdx) {
                llvm::Value *operand = I.getOperand(opIdx);
                if (llvm::Instruction *opInst = llvm::dyn_cast<llvm::Instruction>(operand)) {
                    if (nodeMap.find(opInst) != nodeMap.end()) {
                        DAGNode *opNode = nodeMap[opInst];
                        dag->addEdge(opNode, node);
                    }
                }
            }

        }
    }

    return dag;
}

