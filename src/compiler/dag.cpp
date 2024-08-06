#include "dag.hpp"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>


DAGNode::DAGNode(llvm::Instruction *i, const std::string &res, const std::string &op, 
            const std::vector<std::string> &ops, const std::string &type)
        : inst(i), result(res), operation(op), operands(ops), operandType(type) {}

void DAGNode::addDependency(DAGNode *node) {
    dependencies.push_back(node);
}

void DAGNode::print() const {
    std::cout << "Node: " << operation << "\n";
    std::cout << "Result: " << result << "\n";
    std::cout << "Operands: ";
    for (const auto &operand : operands) {
        std::cout << operand << " ";
    }
    std::cout << "\n";
    std::cout << "Operand Type: " << operandType << "\n";
    std::cout << "\n";
    std::cout << "Dependencies: ";
    for (const auto &dep : dependencies) {
        std::cout << dep->operation << " ";
    }
    std::cout << "\n";
}



DAGNode* DAG::addNode(llvm::Instruction *inst, const std::string &res, const std::string &op,
                     const std::vector<std::string> &ops, const std::string &type) {
        if (nodeMap.find(inst) == nodeMap.end()) {
            DAGNode *node = new DAGNode(inst, res, op, ops, type);
            nodeMap[inst] = node;
            nodes.push_back(node);
            return node;
        }
        return nodeMap[inst];
    }

void DAG::addEdge(DAGNode *from, DAGNode *to) {
    from->addDependency(to);
}

void DAG::print() const {
    for (const auto &node : nodes) {
        node->print();
        std::cout << "\n";
    }
}

void DAG::convert() {
    
}

