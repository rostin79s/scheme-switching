#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

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
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                // Process each instruction
                // Create nodes and edges for the DAG
                // Example: Print the instruction
                I.print(errs());
                errs() << "\n";
            }
        }
    }

    return 0;
}
