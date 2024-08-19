#include "../frontend/dag.hpp"
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>

#include <fstream>


void DAG::generateBackend() {
    std::ofstream IR("../ir.cpp");
    if (!IR.is_open()) {
        std::cerr << "Failed to open ir.cpp for writing\n";
        return;
    }

    IR << "#include \"fhe_operations.hpp\"\n";
    IR << "#include \"<vector>\"\n\n";

    IR << "using namespace CKKS;\n";
    IR << "using namespace TFHE;\n\n";

    // IR << returnType <<name<<"(";
    IR << "FHEdouble* "<<name<<"(";
    bool first = true;
    for (const auto &input : functionInputs) {
        if (!first) IR << ", ";
        // IR << input.second << " " << input.first;
        IR << "FHEdouble*" << " " << input.first;
        first = false;
    }
    IR << ") {\n";

    IR << "\tCKKS_scheme ck;\n";

    // Generate function body
    for (const auto &node : nodes) {
        std::string operation = node->operation;
        std::string result = node->result;
        std::vector<std::string> operands = node->operands;
        std::string type = node->operandType;
        
        type = "FHEdouble*";

        if (operation.find("FHE") == 0) {
            if (operation == "FHEret") {
                IR << "    return " << operands[0] << ";\n";
                continue;
            }
            
            IR << "    " << type << " "<< result << " = " << "ck." <<operation << "(";
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
    IR << "\tCKKS_scheme ck;\n";
    IR << "\tstd::vector<double> inputs = {/* User inputs */};\n";
    IR << "\tstd::vector<"<< "FHEdouble*" << "> encryptedInputs;\n";
    for (const auto &input : functionInputs) {
        IR << "    "<< "FHEdouble*"<< " " << input.first << " = ck.FHEencrypt(inputs[" << input.first << "]);\n";
    }

    IR << "    "<< "FHEdouble*"  << " result = "<<name<<"(";
    first = true;
    for (const auto &input : functionInputs) {
        if (!first) IR << ", ";
        IR << input.first;
        first = false;
    }
    IR << ", " << "ck";
    IR << ");\n";

    IR << "    double finalResult = FHEdecrypt(result);\n";
    IR << "    std::cout << \"Result: \" << finalResult << std::endl;\n";
    IR << "    return 0;\n";
    IR << "}\n";

    IR.close();
}