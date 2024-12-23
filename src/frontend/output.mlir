module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @_Z9min_indexPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = affine.load %arg0[0] : memref<?xi32>
    %1 = affine.for %arg2 = 1 to 10 iter_args(%arg3 = %0) -> (i32) {
      %3 = affine.load %arg0[%arg2] : memref<?xi32>
      %4 = arith.cmpi slt, %3, %arg3 : i32
      %5 = arith.select %4, %3, %arg3 : i32
      affine.yield %5 : i32
    }
    %2 = vector.broadcast %1 : i32 to vector<10xi32>
    affine.for %arg2 = 0 to 10 step 10 {
      %c0_i32 = arith.constant 0 : i32
      %3 = vector.transfer_read %arg0[%arg2], %c0_i32 : memref<?xi32>, vector<10xi32>
      %4 = arith.cmpi eq, %3, %2 : vector<10xi32>
      %5 = arith.extui %4 : vector<10xi1> to vector<10xi32>
      vector.transfer_write %5, %arg1[%arg2] : vector<10xi32>, memref<?xi32>
    }
    return
  }
}

