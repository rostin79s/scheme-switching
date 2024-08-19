#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>

#include "backend.hpp"

void DAG::generateBackend() {
    std::ofstream IR("../ir.cpp");
    if (!IR.is_open()) {
        std::cerr << "Failed to open ir.cpp for writing\n";
        return;
    }

    printHeaders(IR);
    printUserFunction(IR, *this);
    printMainFunction(IR, *this);

    IR.close();
}

// Function to print headers
void printHeaders(std::ofstream& IR) {
    IR << "#include \"fhe_operations.hpp\"\n";
    IR << "#include \"<vector>\"\n\n";

    IR << "using namespace CKKS;\n";
    IR << "using namespace TFHE;\n\n";
}

// Function to print the user-defined function
void printUserFunction(std::ofstream& IR, const DAG& dag) {
    IR << "FHEdouble* " << dag.name << "(";

    bool first = true;
    for (const auto &input : dag.functionInputs) {
        if (!first) IR << ", ";
        IR << "FHEdouble*" << " " << input.first;
        first = false;
    }
    IR << ") {\n";

    IR << "\tCKKS_scheme ck;\n";

    // Generate function body
    for (const auto &node : dag.nodes) {
        std::string operation = node->operation;
        std::string result = node->result;
        std::vector<std::string> operands = node->operands;
        std::string type = "FHEdouble*";

        if (operation.find("FHE") == 0) {
            if (operation == "FHEret") {
                IR << "    return " << operands[0] << ";\n";
                continue;
            }

            IR << "    " << type << " " << result << " = " << "ck." << operation << "(";
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
}

// Function to print the main function
void printMainFunction(std::ofstream& IR, const DAG& dag) {
    IR << "int main() {\n";
    IR << "\tCKKS_scheme ck;\n";
    IR << "\tstd::vector<double> inputs = {/* User inputs */};\n";
    IR << "\tstd::vector<FHEdouble*> encryptedInputs;\n";

    for (const auto &input : dag.functionInputs) {
        IR << "    FHEdouble* " << input.first << " = ck.FHEencrypt(inputs[" << input.first << "]);\n";
    }

    IR << "    FHEdouble* result = " << dag.name << "(";
    bool first = true;
    for (const auto &input : dag.functionInputs) {
        if (!first) IR << ", ";
        IR << input.first;
        first = false;
    }
    IR << ");\n";

    IR << "    double finalResult = FHEdecrypt(result);\n";
    IR << "    std::cout << \"Result: \" << finalResult << std::endl;\n";
    IR << "    return 0;\n";
    IR << "}\n";
}
