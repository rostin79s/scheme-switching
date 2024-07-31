#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include "frontend.hpp"
#include "dag.hpp"




    
int main(int argc, char** argv) {
    llvm::LLVMContext Context;
    llvm::SMDiagnostic Err;

    // Parse the IR file into a module
    std::unique_ptr<llvm::Module> M = llvm::parseIRFile("test.ll", Err, Context);
    if (!M) {
        Err.print(argv[0], llvm::errs());
        return 1;
    }

    // Iterate over the functions in the module
    for (llvm::Function &F : *M) {
        llvm::errs() << "Function: " << F.getName() << "\n";

        printFunctionArguments(F);

        // Build the DAG for the function
        DAG *dag = buildDAGFromInstructions(F);

        // Print the DAG
        dag->print(llvm::errs());

        // Clean up
        delete dag;
    }
    
    return 0;
}
