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
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"



#include <string>
#include <iostream>

#include "frontend/dag.hpp"
#include "frontend/frontend.hpp"
#include "backend/backend.hpp"

using namespace mlir;

mlir::Type convertType(mlir::Type originalType, MLIRContext *context) {
    if (originalType.isF64()) {
        return emitc::OpaqueType::get(context, "FHEdouble*");
    } else if (originalType.isInteger(32)) {
        return emitc::OpaqueType::get(context, "FHEi32*");
    }
    // Return the original type if no conversion is needed
    return originalType;
}

namespace {

  // General pattern for replacing any arithmetic operation with `emitc.call`
template <typename ArithOp>
struct ReplaceArithWithEmitCCallPattern : public OpRewritePattern<ArithOp> {
    using OpRewritePattern<ArithOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ArithOp op, PatternRewriter &rewriter) const override {
        // Map the arithmetic operation type to the corresponding `emitc.call` callee name
        StringRef calleeName;
        if (std::is_same<ArithOp, arith::AddFOp>::value) {
            calleeName = "FHEaddf";
        } else if (std::is_same<ArithOp, arith::MulFOp>::value) {
            calleeName = "FHEmulf";
        } else if (std::is_same<ArithOp, arith::SubFOp>::value) {
            calleeName = "FHEsubf";
        } else if (std::is_same<ArithOp, arith::DivFOp>::value) {
            calleeName = "FHEdivf";
        } else {
            return failure();  // Unsupported operation type
        }

        // Get the operands
        mlir::Value lhs = op.getOperand(0);
        mlir::Value rhs = op.getOperand(1);

        mlir::Type newType = convertType(op.getType(),op.getContext());

        ValueRange operands = {lhs, rhs};

        // Create empty ArrayAttr for args and template_args if not needed
        ArrayAttr argsAttr = rewriter.getArrayAttr({});
        ArrayAttr templateArgsAttr = rewriter.getArrayAttr({});

        // Create the emitc.call operation
        auto callOp = rewriter.create<emitc::CallOp>(
            op.getLoc(),
            newType, // Adjust return type as needed
            calleeName,          // Callee name as StringRef
            argsAttr,            // args as ArrayAttr
            templateArgsAttr,    // template_args as ArrayAttr
            operands             // operands as ValueRange
        );

        outs()<<"callOp: "<<callOp<<"\n";

        // Replace the original operation with the `emitc.call`
        rewriter.replaceOp(op, callOp.getResult(0));

        return success();
    }
};

  

  // Define a pass to apply the patterns for all arithmetic operations.
  struct ReplaceOpsWithEmitCCallPass : public PassWrapper<ReplaceOpsWithEmitCCallPass, OperationPass<ModuleOp>> {
    void runOnOperation() override {
      auto module = getOperation();

        module->removeAttr("llvm.data_layout");
        module->removeAttr("llvm.target_triple");
        module->removeAttr("dlti.dl_spec");
        module->removeAttr("polygeist.target-cpu");
        module->removeAttr("polygeist.target-features");
        module->removeAttr("polygeist.tune-cpu");

        // Set up a pattern rewriter.
        RewritePatternSet patterns(&getContext());

        // Add patterns for arithmetic operations
        patterns.add<ReplaceArithWithEmitCCallPattern<arith::AddFOp>>(&getContext());
        patterns.add<ReplaceArithWithEmitCCallPattern<arith::MulFOp>>(&getContext());
        patterns.add<ReplaceArithWithEmitCCallPattern<arith::SubFOp>>(&getContext());
        patterns.add<ReplaceArithWithEmitCCallPattern<arith::DivFOp>>(&getContext());

        // Apply the patterns greedily.
        if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
            signalPassFailure();
        }
    }
  };
} // end anonymous namespace

int main(int argc, char **argv) {
    // Initialize MLIR context with all dialects.
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::DialectRegistry registry;
    registry.insert<mlir::DLTIDialect>();  // Assuming DLTI dialect is available
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<emitc::EmitCDialect>();

    // Attach the registry to the context
    context.appendDialectRegistry(registry);

    // Open and parse the MLIR file
    std::string filename = "test.mlir";
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(filename, &context);
    if (!module) {
        llvm::errs() << "Error parsing MLIR file\n";
        return 1;
    }

    // Apply the pass to replace operations with EmitC function calls.
    PassManager pm(&context);
    pm.addPass(std::make_unique<ReplaceOpsWithEmitCCallPass>());
    if (failed(pm.run(*module))) {
        llvm::errs() << "Pass failed\n";
        return 1;
    }

    // Output the transformed module
    std::string outputFilename = "ir.mlir";
    auto outputFile = openOutputFile(outputFilename);
    if (!outputFile) {
        llvm::errs() << "Error opening output file\n";
        return 1;
    }

    module->print(outputFile->os());
    outputFile->keep();
    llvm::outs() << "Transformed module saved to: " << outputFilename << "\n";

    // mlir::mlirTranslateMain(argc, argv, "EmitC Translation Tool");

    return 0;
}
