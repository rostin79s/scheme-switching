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


#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"



#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdlib>

#include "frontend/dag.hpp"
#include "frontend/frontend.hpp"
#include "backend/backend.hpp"

using namespace mlir;






namespace {

struct ArithToEmitc : public PassWrapper<ArithToEmitc, OperationPass<ModuleOp>> {
    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder builder(module.getContext());

        auto fhecontext = emitc::OpaqueType::get(builder.getContext(), "CKKS_scheme&");
        auto fhedouble = emitc::OpaqueType::get(builder.getContext(), "FHEdouble");

        // Iterate over functions in the module.
        module.walk([&](func::FuncOp func) {

        builder.setInsertionPoint(func);

        std::string originalName = func.getName().str();
        std::string demangledName = demangle(originalName);
        std::string newName = demangledName;
        func.setName(newName);

        // Update function argument types and return type.
        auto funcType = func.getFunctionType();
        SmallVector<mlir::Type, 4> newInputTypes;

        newInputTypes.push_back(fhecontext);
        for (auto inputType : funcType.getInputs()) {
            // Convert types like `f64` to `emitc.opaque<"FHEdouble">`.
            if (inputType.isF64()) {
            newInputTypes.push_back(fhedouble);
            } else {
            newInputTypes.push_back(inputType);
            }
        }

        // Convert the return type similarly.
        mlir::Type newReturnType = funcType.getResult(0).isF64()
                                ? fhedouble
                                : funcType.getResult(0);

        func.setType(mlir::FunctionType::get(builder.getContext(), newInputTypes, newReturnType));

        Block &entryBlock = func.getBody().front();

        SmallVector<BlockArgument, 4> newArgs;
        // Add new CKKS_scheme& argument at the beginning
        newArgs.push_back(entryBlock.insertArgument(
            entryBlock.args_begin(), 
            newInputTypes[0], 
            builder.getUnknownLoc()
        ));

        for (size_t i = 1; i < newInputTypes.size(); ++i) {
            auto oldArg = entryBlock.getArgument(i);
            if (newInputTypes[i] != oldArg.getType()) {
                oldArg.setType(newInputTypes[i]);
            }
        }


        auto ckarg = func.getArgument(0);
        outs() << "ckarg: " << ckarg << '\n';

        func.walk([&](Operation *op) {
            outs() << "\noperation: " << *op << "\n\n\n";
            
            
            if (auto cmpOp = dyn_cast<arith::CmpFOp>(op)){
                builder.setInsertionPoint(cmpOp);
                auto arg0 = cmpOp.getOperand(0);
                auto arg1 = cmpOp.getOperand(1);
                auto predicate = cmpOp.getPredicate();
                auto s = stringifyCmpFPredicate(predicate);

                outs() << "operation: " << *op << "\n\n\n";

                outs() << "arg0: " << arg0 << '\n';
                outs() << "arg1: " << arg1 << '\n';
                outs() << "predicate: " << predicate << '\n';
                auto newOp = builder.create<emitc::CallOp>(
                    cmpOp.getLoc(),
                    TypeRange(fhedouble),
                    llvm::StringRef("FHE" + s.str() + "f"),
                    ArrayAttr(),
                    ArrayAttr(),
                    mlir::ArrayRef<mlir::Value>{ckarg, arg0, arg1});

                cmpOp.replaceAllUsesWith(newOp.getResult(0));
                cmpOp.erase();
            }
            
            else if (auto selectOp = dyn_cast<arith::SelectOp>(op)){
                builder.setInsertionPoint(selectOp);
                auto arg0 = selectOp.getOperand(0);
                auto arg1 = selectOp.getOperand(1);
                auto arg2 = selectOp.getOperand(2);
                auto newOp = builder.create<emitc::CallOp>(
                    selectOp.getLoc(),
                    TypeRange(fhedouble),
                    llvm::StringRef("FHEselectf"),
                    ArrayAttr(),
                    ArrayAttr(),
                    mlir::ArrayRef<mlir::Value>{ckarg, arg0, arg1, arg2});

                selectOp.replaceAllUsesWith(newOp.getResult(0));
                selectOp.erase();
                
            }

            else if (auto forOp = dyn_cast<scf::ForOp>(op)){
                builder.setInsertionPoint(forOp);
                auto *newblock = forOp.getBody();
                auto res = forOp.getResults();
                for (auto it : llvm::enumerate(res)) {
                    outs() <<"for loop res: " << it.value() << "  type: " << it.value().getType() << "\n";
                    if (it.value().getType().isF64()) {
                        it.value().setType(fhedouble);
                    }
                }
                for (auto it : llvm::enumerate(newblock->getArguments())) {
                    outs() <<"for loop arg: " << it.value() << "  type: " << it.value().getType() << "\n";
                    if (it.value().getType().isF64()) {
                        
                        it.value().setType(fhedouble);
                    }
                }
            }

            else if (auto arithOp = dyn_cast<arith::ConstantFloatOp>(op)){
                outs()<<"float constant"<<"\n";
                builder.setInsertionPoint(arithOp);
                auto value = arithOp.value();
                double value2 = value.convertToDouble();

                auto newConstantOp = builder.create<emitc::ConstantOp>(
                    arithOp.getLoc(),
                    builder.getF64Type(), // Type for the constant
                    builder.getF64FloatAttr(value2) // Create an F64 attribute for the double value
                );

                // std::stringstream ss;
                // ss << value2;
                // auto opaqueAttr = emitc::OpaqueAttr::get(builder.getContext(),ss.str());

                // auto newConstantOp = builder.create<emitc::ConstantOp>(
                //     arithOp.getLoc(),fhedouble,opaqueAttr);

                auto newCallOp = builder.create<emitc::CallOp>(
                    arithOp.getLoc(),
                    TypeRange(fhedouble),
                    llvm::StringRef("FHEencrypt"),
                    ArrayAttr(),
                    ArrayAttr(),
                    mlir::ArrayRef<mlir::Value>{ckarg, newConstantOp.getResult()}
                );
                
                arithOp.replaceAllUsesWith(newCallOp->getResult(0));
                arithOp.erase();          
            }

            else if (auto arithOp = dyn_cast<arith::AddFOp>(op)) {
                builder.setInsertionPoint(arithOp);

                // Create a new `emitc.call` operation.
                auto arg0 = arithOp.getOperand(0);
                auto arg1 = arithOp.getOperand(1);
                outs() << "arg1: " << arg1 << '\n';
                

                auto newOp = builder.create<emitc::CallOp>(
                    arithOp.getLoc(),
                    TypeRange(fhedouble),
                    llvm::StringRef("FHEaddf"),
                    ArrayAttr(),
                    ArrayAttr(),
                    mlir::ArrayRef<mlir::Value>{ckarg, arg0, arg1});

                // Replace the original addf operation with the new one.
                arithOp.replaceAllUsesWith(newOp.getResult(0));
                arithOp.erase();
            }
            else if (auto arithOp = dyn_cast<arith::SubFOp>(op)) {
                builder.setInsertionPoint(arithOp);

                // Create a new `emitc.call` operation.
                auto arg0 = arithOp.getOperand(0);
                auto arg1 = arithOp.getOperand(1);
                

                auto newOp = builder.create<emitc::CallOp>(
                    arithOp.getLoc(),
                    TypeRange(fhedouble),
                    llvm::StringRef("FHEsubf"),
                    ArrayAttr(),
                    ArrayAttr(),
                    mlir::ArrayRef<mlir::Value>{ckarg, arg0, arg1});

                // Replace the original addf operation with the new one.
                arithOp.replaceAllUsesWith(newOp.getResult(0));
                arithOp.erase();
            }

            else if (auto arithOp = dyn_cast<arith::MulFOp>(op)) {
                builder.setInsertionPoint(arithOp);

                // Create a new `emitc.call` operation.
                auto arg0 = arithOp.getOperand(0);
                auto arg1 = arithOp.getOperand(1);
                

                auto newOp = builder.create<emitc::CallOp>(
                    arithOp.getLoc(),
                    TypeRange(fhedouble),
                    llvm::StringRef("FHEmulf"),
                    ArrayAttr(),
                    ArrayAttr(),
                    mlir::ArrayRef<mlir::Value>{ckarg, arg0, arg1});

                // Replace the original addf operation with the new one.
                arithOp.replaceAllUsesWith(newOp.getResult(0));
                arithOp.erase();
            }
            else if (auto arithOp = dyn_cast<arith::DivFOp>(op)) {
                builder.setInsertionPoint(arithOp);

                // Create a new `emitc.call` operation.
                auto arg0 = arithOp.getOperand(0);
                auto arg1 = arithOp.getOperand(1);
                

                auto newOp = builder.create<emitc::CallOp>(
                    arithOp.getLoc(),
                    TypeRange(fhedouble),
                    llvm::StringRef("FHEdivf"),
                    ArrayAttr(),
                    ArrayAttr(),
                    mlir::ArrayRef<mlir::Value>{ckarg, arg0, arg1});

                // Replace the original addf operation with the new one.
                arithOp.replaceAllUsesWith(newOp.getResult(0));
                arithOp.erase();
            }
            
        });
        });
    }
    

}; // end anonymous namespace

}

int main(int argc, char **argv) {
    // Initialize MLIR context with all dialects.
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    mlir::DialectRegistry registry;
    registry.insert<mlir::DLTIDialect>();  // Assuming DLTI dialect is available
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<emitc::EmitCDialect>();
    registry.insert<mlir::scf::SCFDialect>();

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

    outs() << "starting pass\n";
    PassManager pm(&context);
    pm.addPass(std::make_unique<ArithToEmitc>());
    if (failed(pm.run(*module))) {
        llvm::errs() << "Pass failed\n";
        return 1;
    }

    outs() << "Pass succeeded\n";

    // Output the transformed module
    std::string outputFilename = "ir.mlir";
    auto outputFile = openOutputFile(outputFilename);
    if (!outputFile) {
        llvm::errs() << "Error opening output file\n";
        return 1;
    }

    module->print(outputFile->os());
    outputFile->keep();

    return 0;
}
