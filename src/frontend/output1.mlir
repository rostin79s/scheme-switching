module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @_Z6rostinPdS_d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = vector.broadcast %arg2 : f64 to vector<5xf64>
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant dense<0.000000e+00> : vector<5xf64>
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c5_2 = arith.constant 5 : index
    %1 = scf.for %arg3 = %c0 to %c5 step %c5_2 iter_args(%arg4 = %cst_1) -> (vector<5xf64>) {
      %cst_3 = arith.constant dense<0.000000e+00> : vector<5xf64>
      %cst_4 = arith.constant dense<1.000000e+00> : vector<5xf64>
      %cst_5 = arith.constant 0.000000e+00 : f64
      %3 = vector.transfer_read %arg0[%arg3], %cst_5 : memref<?xf64>, vector<5xf64>
      %cst_6 = arith.constant 0.000000e+00 : f64
      %4 = vector.transfer_read %arg1[%arg3], %cst_6 : memref<?xf64>, vector<5xf64>
      %5 = arith.subf %3, %4 : vector<5xf64>
      %6 = arith.mulf %5, %0 : vector<5xf64>
      %7 = arith.cmpf oeq, %6, %0 : vector<5xf64>
      %8 = arith.select %7, %cst_4, %cst_3 : vector<5xi1>, vector<5xf64>
      %9 = arith.addf %arg4, %8 : vector<5xf64>
      scf.yield %9 : vector<5xf64>
    }
    %2 = vector.reduction <add>, %1 : vector<5xf64> into f64
    return %2 : f64
  }
}

