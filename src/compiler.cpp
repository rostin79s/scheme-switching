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






namespace {

    template <typename OpType>
    class EmitCBasicPattern final : public OpConversionPattern<OpType>{
    protected:
        using OpConversionPattern<OpType>::typeConverter;

    public:
        using OpConversionPattern<OpType>::OpConversionPattern;

        LogicalResult matchAndRewrite(
            OpType op, typename OpType::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override{

            rewriter.setInsertionPoint(op);
            auto dstType = typeConverter->convertType(op.getType());
            outs() << "dstType: " << dstType << "\n";
            if (!dstType)
                return failure();

            // Materialize the operands where necessary
            llvm::SmallVector<mlir::Value> materialized_operands;
            for (mlir::Value o : op.getOperands())
            {
                
                auto operandDstType = typeConverter->convertType(o.getType());
                outs() << "operandDstType: " << operandDstType << "\n";
                if (!operandDstType)
                    return failure();
                if (o.getType() != operandDstType)
                {
                    auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                    outs() << "new_operand: " << new_operand << "\n";
                    materialized_operands.push_back(new_operand);
                }
                else
                {
                    materialized_operands.push_back(o);
                }
            }

            // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with
            // seal's API)
            std::string op_str = "";
            outs() << -123 << "\n";
            if (std::is_same<OpType, arith::AddFOp>()){
                op_str = "FHEaddf";
                outs() << -1234 << "\n";
            }
            else if (std::is_same<OpType, arith::SubFOp>())
                op_str = "FHEsubf";
            else if (std::is_same<OpType, arith::MulFOp>())
                op_str = "FHEmulf";
            else if (std::is_same<OpType, arith::DivFOp>())
                op_str = "FHEdivf";
            else if (std::is_same<OpType, arith::AddIOp>())
                op_str = "FHEadd";
            else if (std::is_same<OpType, arith::SubIOp>())
                op_str = "FHEsub";
            else if (std::is_same<OpType, arith::MulIOp>())
                op_str = "FHEmul";
            else
                return failure();

                
            outs() << "old operation was : " << op << "name: " << op_str << "\n";
            auto newop = rewriter.create<emitc::CallOp>(
                op.getLoc(), TypeRange(dstType), llvm::StringRef(op_str), ArrayAttr(), ArrayAttr(),
                materialized_operands);

            outs() << "new operation is : " << newop << "\n";
            rewriter.replaceOp(op,newop);
            return success();
        }
    };


    class FunctionConversionPattern final : public OpConversionPattern<func::FuncOp>{
    public:
        using OpConversionPattern<func::FuncOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(
            func::FuncOp op, typename func::FuncOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override{
            // Compute the new signature of the function.
            TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
            SmallVector<mlir::Type> newResultTypes;
            if (failed(typeConverter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
                return failure();
            if (typeConverter->convertSignatureArgs(op.getFunctionType().getInputs(), signatureConversion).failed())
                return failure();
            
            auto new_functype = mlir::FunctionType::get(getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

            rewriter.startRootUpdate(op);
            op.setType(new_functype);
            for (auto it = op.getRegion().args_begin(); it != op.getRegion().args_end(); ++it)
            {
                auto arg = *it;
                auto oldType = arg.getType();
                auto newType = typeConverter->convertType(oldType);
                arg.setType(newType);
                if (newType != oldType)
                {
                    rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
                    auto m_op = typeConverter->materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
                    arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
                }
            }
            rewriter.finalizeRootUpdate(op);

            outs() << op << "\n";
            return success();
        }
    };

    class EmitCReturnPattern final : public OpConversionPattern<func::ReturnOp>{
    public:
        using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
        
        LogicalResult matchAndRewrite(
            func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override{
            if (op->getNumOperands() != 1)
            {
                emitError(op->getLoc(), "Only single value returns support for now.");
                return failure();
            }
            auto dstType = this->getTypeConverter()->convertType(op->getOperandTypes().front());
            if (!dstType)
                return failure();
            if (auto bst = dstType.dyn_cast_or_null<emitc::OpaqueType>())
            {
                rewriter.setInsertionPoint(op);
                auto materialized =
                    typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.getOperands());
                // build a new return op
                rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

            } // else do nothing
            return success();
        }
    };

    struct ReplaceOpsWithEmitCCallPass : public PassWrapper<ReplaceOpsWithEmitCCallPass, OperationPass<ModuleOp>> {
        void runOnOperation() override {
        // TODO: We still need to emit a pre-amble with an include statement
        //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

        auto type_converter = TypeConverter();

        type_converter.addConversion([&](mlir::Type t) {
            if (t.isF64())
                return std::optional<mlir::Type>(emitc::OpaqueType::get(&getContext(), "FHEdouble*"));
            else if (t.isInteger(32))
                return std::optional<mlir::Type>(emitc::OpaqueType::get(&getContext(), "FHEint*"));
            else
                return std::optional<mlir::Type>(t);
        });
        type_converter.addTargetMaterialization([&](OpBuilder &builder, mlir::Type t, ValueRange vs, Location loc) {
            if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
            {
                assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
                auto old_type = vs.front().getType();
                if (old_type.isF64())
                {
                    if (ot.getValue().str() == "FHEdouble*")
                        // return std::optional<mlir::Value>(builder.create<mlir::Value>(loc, ot, vs));
                        return std::optional<mlir::Value>(vs.front());
                }
                else if (old_type.isInteger(32))
                {
                    if (ot.getValue().str() == "FHEint*")
                        // return std::optional<mlir::Value>(builder.create<mlir::Value>(loc, ot, vs));
                        return std::optional<mlir::Value>(vs.front());
                }
            }
            return std::optional<mlir::Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
        });
        type_converter.addArgumentMaterialization([&](OpBuilder &builder, mlir::Type t, ValueRange vs, Location loc) {
            if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>()){
                assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
                auto old_type = vs.front().getType();

                if (old_type.isF64()) {
                    if (ot.getValue().str() == "FHEdouble*") {
                        // return std::optional<mlir::Value>(builder.create<mlir::Value>(loc, ot, vs));
                        return std::optional<mlir::Value>(vs.front());
                    }
                }
                else if (old_type.isInteger(32)) {
                    if (ot.getValue().str() == "FHEint*") {
                        // return std::optional<mlir::Value>(builder.create<mlir::Value>(loc, ot, vs));
                        return std::optional<mlir::Value>(vs.front());
                    }
                }
            }
            return std::optional<mlir::Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
        });

        type_converter.addSourceMaterialization([&](OpBuilder &builder, mlir::Type t, ValueRange vs, Location loc) {
            if (t.isF64()){
                assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
                auto old_type = vs.front().getType();
                if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                    if (ot.getValue().str() == "FHEdouble*")

                        // return std::optional<mlir::Value>(builder.create<>(loc, ot, vs));
                        return std::optional<mlir::Value>(vs.front());
            }
            else if (t.isInteger(32)){
                assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
                auto old_type = vs.front().getType();
                if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                    if (ot.getValue().str() == "FHEint*")
                        // return std::optional<mlir::Value>(builder.create<mlir::Value>(loc, ot, vs));
                        return std::optional<mlir::Value>(vs.front());
            }

            return std::optional<mlir::Value>(std::nullopt);
        });
        // auto module = getOperation();
        // module->removeAttr("llvm.data_layout");
        // module->removeAttr("llvm.target_triple");
        // module->removeAttr("dlti.dl_spec");
        // module->removeAttr("polygeist.target-cpu");
        // module->removeAttr("polygeist.target-features");
        // module->removeAttr("polygeist.tune-cpu");

        ConversionTarget target(getContext());
        target.addLegalDialect<emitc::EmitCDialect>();
        // target.addLegalDialect<mlir::func::FuncDialect>();
        target.addLegalOp<ModuleOp>();
        target.addLegalOp<func::FuncOp>();
        target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
            auto fop = llvm::dyn_cast<func::FuncOp>(op);
            for (auto t : op->getOperandTypes())
            {
                if (!type_converter.isLegal(t))
                    return false;
            }
            for (auto t : op->getResultTypes())
            {
                if (!type_converter.isLegal(t))
                    return false;
            }
            for (auto t : fop.getFunctionType().getInputs())
            {
                if (!type_converter.isLegal(t))
                    return false;
            }
            for (auto t : fop.getFunctionType().getResults())
            {
                if (!type_converter.isLegal(t))
                    return false;
            }
            return true;
        });
        target.addDynamicallyLegalOp<func::ReturnOp>(
            [&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });
        mlir::RewritePatternSet patterns(&getContext());

        // patterns.add<
        //     EmitCBasicPattern<arith::AddFOp>, EmitCBasicPattern<arith::MulFOp>, EmitCBasicPattern<arith::SubFOp>,
        //     EmitCBasicPattern<arith::DivFOp>, FunctionConversionPattern, EmitCReturnPattern>(
        //     type_converter, patterns.getContext());

        patterns.add<
            FunctionConversionPattern, EmitCReturnPattern>(
            type_converter, patterns.getContext());
        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
            // outs() << getOperation() << "\n";
            signalPassFailure();
        
        }
        

    }; // end anonymous namespace

}

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

    outs() << "starting pass\n";
    PassManager pm(&context);
    pm.addPass(std::make_unique<ReplaceOpsWithEmitCCallPass>());
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

    // mlir::mlirTranslateMain(argc, argv, "EmitC Translation Tool");

    return 0;
}
