#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>
#include <unordered_map>
#include <vector>
#include <set>

using namespace llvm;

// Class to represent a node in the DAG
class DAGNode {
public:
    Instruction *inst;
    std::string result;
    std::string operation;
    std::vector<std::string> operands;
    std::vector<std::string> operandTypes;
    std::vector<DAGNode*> dependencies; // List of nodes that this node depends on

    DAGNode(Instruction *i, const std::string &res, const std::string &op, 
            const std::vector<std::string> &ops, const std::vector<std::string> &types)
        : inst(i), result(res), operation(op), operands(ops), operandTypes(types) {}

    void addDependency(DAGNode *node) {
        dependencies.push_back(node);
    }

    void print(raw_ostream &OS) const {
        OS << "Node: " << operation << "\n";
        OS << "Result: " << result << "\n";
        OS << "Operands: ";
        for (const auto &operand : operands) {
            OS << operand << " ";
        }
        OS << "\n";
        OS << "Operand Types: ";
        for (const auto &type : operandTypes) {
            OS << type << " ";
        }
        OS << "\n";
        OS << "Dependencies: ";
        for (const auto &dep : dependencies) {
            OS << dep->operation << " ";
        }
        OS << "\n";
    }
};

// Class to represent the Directed Acyclic Graph (DAG)
class DAG {
public:
    std::unordered_map<Instruction*, DAGNode*> nodeMap;
    std::vector<DAGNode*> nodes;
    DAGNode* addNode(Instruction *inst, const std::string &res, const std::string &op,
                     const std::vector<std::string> &ops, const std::vector<std::string> &types) {
        if (nodeMap.find(inst) == nodeMap.end()) {
            DAGNode *node = new DAGNode(inst, res, op, ops, types);
            nodeMap[inst] = node;
            nodes.push_back(node);
            return node;
        }
        return nodeMap[inst];
    }

    void addEdge(DAGNode *from, DAGNode *to) {
        from->addDependency(to);
    }

    void print(raw_ostream &OS) const {
        for (const auto &node : nodes) {
            node->print(OS);
            OS << "\n";
        }
    }
};

void printFunctionArguments(Function &F) {
    errs() << "Function: " << F.getName() << "\n";

    for (auto &Arg : F.args()) {
        // Print the name of the parameter
        Value *result = &Arg; // The result is the instruction itself
        std::string resultStr;
        raw_string_ostream rso(resultStr);
        result->printAsOperand(rso, false);

        // Print the type of the parameter
        std::string paramType;
        raw_string_ostream rsoType(paramType);
        Arg.getType()->print(rsoType);

        // Output parameter information
        errs() << "Parameter: " << resultStr << ", Type: " << rsoType.str() << "\n";
    }
}

// Function to process each instruction and build the DAG
DAG* buildDAGFromInstructions(Function &F) {

    printFunctionArguments(F);

    DAG *dag = new DAG();

    // Iterate over instructions and build nodes and edges
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            // Ignore allocation instructions
            if (isa<AllocaInst>(&I)) {
                continue;
            }

            // Extract result, operation, operands, and operand types
            Value *result = &I; // The result is the instruction itself
            std::string resultStr;
            raw_string_ostream rso(resultStr);
            result->printAsOperand(rso, false);

            std::string operationStr = I.getOpcodeName();
            std::vector<std::string> operandStrs;
            std::vector<std::string> operandTypes;

            for (unsigned int opIdx = 0; opIdx < I.getNumOperands(); ++opIdx) {
                Value *operand = I.getOperand(opIdx);

                // Convert operand to string
                std::string operandStr;
                raw_string_ostream rsoOperand(operandStr);
                operand->printAsOperand(rsoOperand, false);
                operandStrs.push_back(rsoOperand.str());

                // Convert operand type to string
                Type *type = operand->getType();
                std::string typeStr;
                raw_string_ostream rstoType(typeStr);
                type->print(rstoType);
                operandTypes.push_back(rstoType.str());
            }

            // Create a node for the instruction
            DAGNode *node = dag->addNode(&I, resultStr, operationStr, operandStrs, operandTypes);

            // Create edges based on operand dependencies
            for (unsigned int opIdx = 0; opIdx < I.getNumOperands(); ++opIdx) {
                Value *operand = I.getOperand(opIdx);
                if (Instruction *opInst = dyn_cast<Instruction>(operand)) {
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



int main(int argc, char** argv) {
    LLVMContext Context;
    SMDiagnostic Err;

    // Parse the IR file into a module
    std::unique_ptr<Module> M = parseIRFile("test.ll", Err, Context);
    if (!M) {
        Err.print(argv[0], errs());
        return 1;
    }

    // Iterate over the functions in the module
    for (Function &F : *M) {
        errs() << "Function: " << F.getName() << "\n";

        // Build the DAG for the function
        DAG *dag = buildDAGFromInstructions(F);

        // Print the DAG
        dag->print(errs());

        // Clean up
        delete dag;
    }

    return 0;
}
