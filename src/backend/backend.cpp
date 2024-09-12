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
#include "backend.hpp"

void DAG::generateBackend(llvm::LLVMContext &Context) {
    llvm::Module module("FHE_module", Context);  // Declared within the function
    llvm::IRBuilder<> builder(Context);

    // Example FHE types - adjust these according to your specific FHE backend types
    llvm::StructType *fheIntTy = llvm::StructType::create(Context, "FHEdouble");  // FHE encrypted int type named FHEi32
    llvm::StructType *fheDoubleTy = llvm::StructType::create(Context, "FHEdouble");  // FHE encrypted double type named FHEdouble

    // Determine the return type of the function based on `returnType`
    llvm::Type *llvmReturnType = nullptr;
    if (returnType == "FHEi32") {
        llvmReturnType = fheIntTy;
    } else if (returnType == "FHEdouble") {
        llvmReturnType = fheDoubleTy;
    } else {
        llvm::errs() << "Unknown return type: " << returnType << "\n";
        return;
    }

    // Prepare function argument types and names based on `functionInputs`
    std::vector<llvm::Type*> argTypes;
    std::vector<std::string> argNames;
    for (const auto &input : functionInputs) {
        const std::string &argName = input.first;
        const std::string &argType = input.second;

        llvm::Type *llvmArgType = nullptr;
        if (argType == "FHEi32") {
            llvmArgType = fheIntTy;
        } else if (argType == "FHEdouble") {
            llvmArgType = fheDoubleTy;
        } else {
            llvm::errs() << "Unknown argument type: " << argType << " for argument " << argName << "\n";
            return;
        }

        argTypes.push_back(llvmArgType);
        argNames.push_back(argName);
    }

    // Create the function type
    llvm::FunctionType *funcType = llvm::FunctionType::get(llvmReturnType, argTypes, false);

    // Create the function using the name stored in `name`
    llvm::Function *function = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name, module);

    // Set function argument names
    unsigned idx = 0;
    for (llvm::Argument &arg : function->args()) {
        arg.setName(argNames[idx++]);
    }

    // Create the entry block for the function
    llvm::BasicBlock *entryBlock = llvm::BasicBlock::Create(Context, "entry", function);
    builder.SetInsertPoint(entryBlock);

    // Map to store generated values
    std::unordered_map<std::string, llvm::Value*> valueMap;

    // Populate the value map with function arguments
    idx = 0;
    for (llvm::Argument &arg : function->args()) {
        valueMap[arg.getName().str()] = &arg;
    }

    // Iterate through the DAG and generate function calls
    llvm::Value *returnValue;
    for (DAGNode *node : nodes) {
        if (node->operation == "FHEret"){
            returnValue = valueMap[node->operands[0]];
            continue;
        }

        llvm::Function *callee = module.getFunction(node->operation);

        // If the function isn't declared yet, declare it
        if (!callee) {
            llvm::Type *firstOperandType = nullptr;
            llvm::Type *secondOperandType = nullptr;

            // Determine the types of the first and second operands
            if (node->operands.size() >= 1) {
                if (functionInputs.find(node->operands[0]) != functionInputs.end()) {
                    if (functionInputs[node->operands[0]] == "FHEi32") {
                        firstOperandType = fheIntTy;
                    } else if (functionInputs[node->operands[0]] == "FHEdouble") {
                        firstOperandType = fheDoubleTy;
                    }
                } else if (node->operands[0].find_first_not_of("-0123456789") == std::string::npos) {
                    // First operand is a plaintext integer constant
                    firstOperandType = llvm::Type::getInt32Ty(Context);
                }
            }

            if (node->operands.size() >= 2) {
                if (functionInputs.find(node->operands[1]) != functionInputs.end()) {
                    if (functionInputs[node->operands[1]] == "FHEi32") {
                        secondOperandType = fheIntTy;
                    } else if (functionInputs[node->operands[1]] == "FHEdouble") {
                        secondOperandType = fheDoubleTy;
                    }
                } else if (node->operands[1].find_first_not_of("-0123456789") == std::string::npos) {
                    // Second operand is a plaintext integer constant
                    secondOperandType = llvm::Type::getInt32Ty(Context);
                }
            }

            // Fallback to FHE types if not plaintext
            if (!firstOperandType) {
                firstOperandType = fheIntTy;  // Default FHE type
            }
            if (!secondOperandType) {
                secondOperandType = fheIntTy;  // Default FHE type
            }

            // Create the function type based on the operand types
            llvm::FunctionType *calleeFuncType = llvm::FunctionType::get(fheIntTy, {firstOperandType, secondOperandType}, false);

            // Create the function
            callee = llvm::Function::Create(calleeFuncType, llvm::Function::ExternalLinkage, node->operation, module);
        }   

        // Convert operands to LLVM values
        std::vector<llvm::Value*> llvmOperands;
        for (const std::string &operandName : node->operands) {
            llvm::Value *operandValue = nullptr;

            // Check if operandName is a numeric constant
            if (operandName.find_first_not_of("-0123456789") == std::string::npos) {
                // It's a numeric constant, create a constant value
                int intValue = std::stoi(operandName);  // Convert string to integer
                operandValue = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), intValue);
            } else {
                // It's a variable, try to retrieve it from the map
                operandValue = valueMap[operandName];
                
                // Handle the case where the operand is not found
                if (!operandValue) {
                    // Check if the operand is a plaintext value
                    if (functionInputs.find(operandName) != functionInputs.end()) {
                        // Create a default value for plaintext types
                        std::string operandType = functionInputs[operandName];
                        llvm::Type *type = nullptr;

                        // Map operandType to LLVM types
                        if (operandType == "FHEi32") {
                            type = llvm::Type::getInt32Ty(Context);
                        } else if (operandType == "FHEdouble") {
                            type = llvm::Type::getDoubleTy(Context);
                        } else {
                            llvm::errs() << "Error: Unknown operand type " << operandType << "\n";
                            return;
                        }

                        // Create a default value (e.g., zero) for the operand
                        operandValue = llvm::Constant::getNullValue(type);
                    } else {
                        llvm::errs() << "Error: Operand " << operandName << " not found.\n";
                        return;
                    }
                }
            }
            
            llvmOperands.push_back(operandValue);
        }

        // Create the function call instruction
        llvm::Value *result = builder.CreateCall(callee, llvmOperands, node->result);

        // Store the result in the map
        valueMap[node->result] = result;
    }

    // Return the appropriate value based on the return type
    if (llvmReturnType != llvm::Type::getVoidTy(Context)) {
        builder.CreateRet(returnValue);
    } else {
        builder.CreateRetVoid();  // Return void for void functions
    }

    // Generate the main function
    generateMainFunction(Context, module);
    // Verify the generated module
    llvm::verifyModule(module, &llvm::errs());

    // Write the module to a file
    std::error_code EC;
    llvm::raw_fd_ostream outFile("output.ll", EC, llvm::sys::fs::OF_None);
    if (EC) {
        llvm::errs() << "Error opening file for writing: " << EC.message() << "\n";
        return;
    }

    module.print(outFile, nullptr);  // Print the module to the file
    outFile.flush();  // Ensure the output is written to the file
}


void generateMainFunction(llvm::LLVMContext &Context, llvm::Module &module) {
    llvm::IRBuilder<> builder(Context);

    // Create a function type (int main())
    llvm::FunctionType *funcType = llvm::FunctionType::get(builder.getInt32Ty(), false);

    // Create the main function with external linkage
    llvm::Function *mainFunction = llvm::Function::Create(
        funcType, llvm::Function::ExternalLinkage, "main", module);

    // Create a basic block and set the insertion point
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(Context, "entry", mainFunction);
    builder.SetInsertPoint(entry);

    // Create a return instruction (return 0)
    builder.CreateRet(builder.getInt32(0));

    // Verify the function to check for errors
    if (llvm::verifyFunction(*mainFunction)) {
        llvm::errs() << "Function verification failed!\n";
        return;
    }
}