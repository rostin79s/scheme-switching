#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>  // For file output
#include <unordered_map>
#include <string>
#include <cstdlib>  // For std::stoi
#include <fstream>
#include "backend.hpp"




void generateCPP(const DAG& dag){
    std::ofstream IR("../ir.cpp");
    if (!IR.is_open()) {
        std::cerr << "Failed to open ir.cpp for writing\n";
        return;
    }

    printHeaders(IR);
    printUserFunction(IR, dag);
    printMainFunction(IR, dag);

    IR.close();

}

// Function to print headers
void printHeaders(std::ofstream& IR) {
    IR << "#include \"src/backend/fhe_operations.hpp\"\n";
    IR << "#include \"src/backend/fhe_types.hpp\"\n";
    IR << "#include <vector>\n";
    IR << "#include <iostream>\n\n";
    IR << "using namespace CKKS;\n";
    IR << "using namespace CGGI;\n\n";
}

// Function to print the user-defined function
void printUserFunction(std::ofstream& IR, const DAG& dag) {
    // Add ck as the first argument
    IR << "FHEdouble* " << dag.name << "(CKKS_scheme& ck";

    // Continue with the other function inputs
    for (const auto &input : dag.functionInputs) {
        IR << ", FHEdouble* " << input.first;
    }
    IR << ") {\n";

    // Generate function body
    for (const auto &node : dag.nodes) {
        std::string operation = node->operation;
        std::string result = node->result;
        std::vector<std::string> operands = node->operands;
        std::string type = "FHEdouble*";

        if (operation.find("FHE") == 0) {
            if (operation == "FHEreturn") {
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
    IR << "\tCKKS_scheme ck(2,50,1);\n";


    int count = 1;
    for (const auto &input : dag.functionInputs) {
        std::string vectorName = "input" + std::to_string(count);
        IR << "\tstd::vector<double> " << vectorName << " = {0};\n";
        IR << "\tFHEdouble* " << input.first << " = ck.FHEencrypt(" << "ck.FHEencode(" << vectorName << "));\n";
        count++;
    }

    // Pass ck as the first argument to the user-defined function
    IR << "    FHEdouble* result = " << dag.name << "(ck";
    for (const auto &input : dag.functionInputs) {
        IR << ", " << input.first;
    }
    IR << ");\n";
    // Decrypt and print the result
    IR << "    FHEplain* res = ck.FHEdecrypt(result);\n";
    IR << "    std::cout << \"Result: \" << res->getPlaintext() << std::endl;\n";
    IR << "    return 0;\n";
    IR << "}\n";
}