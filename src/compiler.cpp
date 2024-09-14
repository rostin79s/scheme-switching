#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <string>
#include <iostream>

int main(int argc, char** argv) {
    // Create an MLIR context
    mlir::MLIRContext context;

    mlir::DialectRegistry registry;
    // registry.insert<mlir::func::FuncDialect>();
    // registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::DLTIDialect>();  // Assuming DLTI dialect is available
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();

    // Attach the registry to the context
    context.appendDialectRegistry(registry);

    // Open and parse the MLIR file
    std::string filename = "test.mlir";
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(filename, &context);
    if (!module) {
        llvm::errs() << "Error parsing MLIR file\n";
        return 1;
    }

    // Iterate over all operations in the module
    module->walk([](mlir::Operation *op) {
        // Print the operation name
        llvm::outs() << "Operation: " << op->getName() << "\n";

        // Print operation location
        llvm::outs() << "  Location: " << op->getLoc() << "\n";

        // Print operands
        for (auto operand : op->getOperands()) {
            llvm::outs() << "  Operand: " << operand << "\n";
        }
    });

    return 0;
}
