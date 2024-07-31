#ifndef DAG_HPP
#define DAG_HPP

#include <llvm/IR/Instruction.h>
#include <llvm/Support/raw_ostream.h>
#include <unordered_map>
#include <vector>

class DAGNode {
public:
    llvm::Instruction *inst;
    std::string result;
    std::string operation;
    std::vector<std::string> operands;
    std::vector<std::string> operandTypes;
    std::vector<DAGNode*> dependencies; // List of nodes that this node depends on

    DAGNode(llvm::Instruction *i, const std::string &res, const std::string &op, 
            const std::vector<std::string> &ops, const std::vector<std::string> &types);

    void addDependency(DAGNode *node);

    void print(llvm::raw_ostream &OS) const;
};

// Class to represent the Directed Acyclic Graph (DAG)
class DAG {
public:
    std::unordered_map<llvm::Instruction*, DAGNode*> nodeMap;
    std::vector<DAGNode*> nodes;
    DAGNode* addNode(llvm::Instruction *inst, const std::string &res, const std::string &op,
                     const std::vector<std::string> &ops, const std::vector<std::string> &types);

    void addEdge(DAGNode *from, DAGNode *to);
    void print(llvm::raw_ostream &OS) const;
};

DAG* buildDAGFromInstructions(llvm::Function &F);

#endif