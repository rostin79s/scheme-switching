#include "dag.hpp"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>


DAGNode::DAGNode(const std::string &res, const std::string &op, 
            const std::vector<std::string> &ops, const std::string &type)
        :result(res), operation(op), operands(ops), operandType(type) {}

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
    std::cout << "Operand Types: " << operandType<<"\n";
    std::cout << "\n";
    std::cout << "Dependencies: ";
    for (const auto &dep : dependencies) {
        std::cout << dep->operation << " ";
    }
    std::cout << "\n";
}

void DAG::setFunctionInputs(const std::unordered_map<std::string, std::string> &inputs) {
        functionInputs = inputs;
}

DAGNode* DAG::addNode(const std::string &res, const std::string &op,
                     const std::vector<std::string> &ops, const std::string &type) {
        DAGNode *node = new DAGNode(res, op, ops, type);
        nodes.push_back(node);
        return node;
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

