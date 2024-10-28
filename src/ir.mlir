module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @rostin(%arg0: !emitc.opaque<"CKKS_scheme&">, %arg1: !emitc.opaque<"FHEdouble">, %arg2: !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble"> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "emitc.constant"() <{value = #emitc.opaque<"1">}> : () -> !emitc.opaque<"FHEdouble">
    %1 = "emitc.constant"() <{value = #emitc.opaque<"0">}> : () -> !emitc.opaque<"FHEdouble">
    %2:3 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %1, %arg5 = %arg2, %arg6 = %arg1) -> (!emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) {
      %3 = emitc.call "FHEaddf"(%arg0, %arg6, %arg5) : (!emitc.opaque<"CKKS_scheme&">, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      %4 = emitc.call "FHEoeqf"(%3, %arg5) : (!emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      %5 = emitc.call "FHEselectf"(%4, %0, %arg4) : (!emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      %6 = emitc.call "FHEaddf"(%arg0, %arg5, %5) : (!emitc.opaque<"CKKS_scheme&">, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      scf.yield %5, %6, %3 : !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">
    }
    return %2#1 : !emitc.opaque<"FHEdouble">
  }
}
