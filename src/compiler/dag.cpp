#include "dag.hpp"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>

#include <fstream>


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
    std::unordered_map<std::string, std::string> ciphertexts = functionInputs;

    for (DAGNode *node : nodes) {
        std::string op = node->operation;
        std::vector<std::string> ops = node->operands;
        std::string type = node->operandType;

        bool isCiphertextOperation = false;
        bool hasPlaintext = false;

        // Determine if the operation should be a FHE operation based on its operands
        for (const auto &operand : ops) {
            std::cout<<operand;
            if (ciphertexts.find(operand) != ciphertexts.end()) {
                isCiphertextOperation = true;
            } else {
                hasPlaintext = true;
            }
        }

        // Convert the operation name accordingly
        if (isCiphertextOperation) {
            if (hasPlaintext) {
                op = "FHE" + op + "P";
            } else {
                op = "FHE" + op;
            }
            node->operation = op;
        }
        ciphertexts[node->result] = node->operandType;
    }
}

void DAG::optimize() {

}

void DAG::sort() {

}

void DAG::generateBackend() {
    std::ofstream IR("../backend/ir.cpp");
    if (!IR.is_open()) {
        std::cerr << "Failed to open ir.cpp for writing\n";
        return;
    }

    IR << "#include \"fhe_operations.hpp\"\n\n";

    // Function declaration
    IR << "void myFHEFunction(";
    bool first = true;
    for (const auto &input : functionInputs) {
        if (!first) IR << ", ";
        IR << "Ciphertext " << input.first;
        first = false;
    }
    IR << ") {\n";

    // Generate function body
    for (const auto &node : nodes) {
        std::string operation = node->operation;
        std::string result = node->result;
        std::vector<std::string> operands = node->operands;

        if (operation.find("FHE") == 0) {
            if (operation == "FHEret") {
                IR << "    return " << operands[0] << ";\n";
                continue;
            }
            
            IR << "    " << result << " = " << operation << "(";
            for (size_t i = 0; i < operands.size(); ++i) {
                IR << operands[i];
                if (i < operands.size() - 1) {
                    IR << ", ";
                }
            }
            IR << ");\n";
        }
    }

    IR << "}\n\n";

    // Create main function
    IR << "int main() {\n";
    IR << "    std::vector<double> inputs = {/* User inputs */};\n";
    IR << "    std::vector<Ciphertext> encryptedInputs;\n";
    for (const auto &input : functionInputs) {
        IR << "    Ciphertext " << input.first << " = FHEencrypt(inputs[" << input.first << "]);\n";
    }

    IR << "    Ciphertext result = myFHEFunction(";
    first = true;
    for (const auto &input : functionInputs) {
        if (!first) IR << ", ";
        IR << input.first;
        first = false;
    }
    IR << ");\n";

    IR << "    double finalResult = FHEdecrypt(result);\n";
    IR << "    std::cout << \"Result: \" << finalResult << std::endl;\n";
    IR << "    return 0;\n";
    IR << "}\n";

    IR.close();
}