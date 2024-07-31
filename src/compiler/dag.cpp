#include "dag.hpp"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>


DAGNode::DAGNode(llvm::Instruction *i, const std::string &res, const std::string &op, 
            const std::vector<std::string> &ops, const std::vector<std::string> &types)
        : inst(i), result(res), operation(op), operands(ops), operandTypes(types) {}

void DAGNode::addDependency(DAGNode *node) {
    dependencies.push_back(node);
}

void DAGNode::print(llvm::raw_ostream &OS) const {
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



DAGNode* DAG::addNode(llvm::Instruction *inst, const std::string &res, const std::string &op,
                     const std::vector<std::string> &ops, const std::vector<std::string> &types) {
        if (nodeMap.find(inst) == nodeMap.end()) {
            DAGNode *node = new DAGNode(inst, res, op, ops, types);
            nodeMap[inst] = node;
            nodes.push_back(node);
            return node;
        }
        return nodeMap[inst];
    }

void DAG::addEdge(DAGNode *from, DAGNode *to) {
    from->addDependency(to);
}

void DAG::print(llvm::raw_ostream &OS) const {
    for (const auto &node : nodes) {
        node->print(OS);
        OS << "\n";
    }
}

