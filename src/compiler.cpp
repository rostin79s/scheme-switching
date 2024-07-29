#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

class InstructionInfo {
public:
    std::string result;
    std::string operation;
    std::vector<std::string> operands;
    std::vector<std::string> operandTypes;
    void setResult(const std::string &res) {
        result = res;
    }

    void setOperation(const std::string &op) {
        operation = op;
    }

    void addOperand(const std::string &operand) {
        operands.push_back(operand);
    }

    void addOperandType(const std::string &type) {
        operandTypes.push_back(type);
    }

    void print(raw_ostream &OS) const {
        OS << "Result: " << result << "\n";
        OS << "Operation: " << operation << "\n";
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
    }
};

InstructionInfo processInstruction(Instruction &I) {
    InstructionInfo info;

    // Extract the result (left-hand side of the assignment)
    Value *result = &I; // The result is the instruction itself
    std::string resultStr;
    raw_string_ostream rso(resultStr);
    result->printAsOperand(rso, false);
    info.setResult(rso.str());


    // Extract the operation
    info.setOperation(I.getOpcodeName());

    if (info.operation == "ret"){
        info.setResult("");
    }

    // Extract operands and their types
    for (unsigned int opIdx = 0; opIdx < I.getNumOperands(); ++opIdx) {
        Value *operand = I.getOperand(opIdx);

        // Convert operand to string
        std::string operandStr;
        raw_string_ostream rsoOperand(operandStr);
        operand->printAsOperand(rsoOperand, false);
        info.addOperand(rsoOperand.str());

        // Convert operand type to string
        Type *type = operand->getType();
        std::string typeStr;
        raw_string_ostream rstoType(typeStr);
        type->print(rstoType);
        info.addOperandType(rstoType.str());
    }

    return info;
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

        for (BasicBlock &BB : F) {
            errs() << "BasicBlock: " << BB.getName() << "\n";

            for (Instruction &I : BB) {
                // Ignore allocation instructions
                if (isa<AllocaInst>(&I)) {
                    continue;
                }

                // Process each instruction and store its information
                errs() << I << "\n";
                InstructionInfo info = processInstruction(I);
                info.print(errs());
                errs() << "\n";
            }
        }
    }

    return 0;
}
