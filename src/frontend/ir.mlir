module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @rostin(%arg0: !emitc.ptr<!emitc.opaque<"FHEcontext">>, %arg1: !emitc.opaque<"FHEdouble">, %arg2: !emitc.opaque<"FHEdouble">, %arg3: !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble"> attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = "emitc.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
    %1 = emitc.call "FHEencrypt"(%arg0, %0) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, f64) -> !emitc.opaque<"FHEdouble">
    %2 = "emitc.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
    %3 = emitc.call "FHEencrypt"(%arg0, %2) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, f64) -> !emitc.opaque<"FHEdouble">
    %4 = "emitc.constant"() <{value = #emitc.opaque<"std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}">}> : () -> !emitc.opaque<"std::vector<double>">
    %5 = emitc.call "FHEencrypt"(%arg0, %4) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"std::vector<double>">) -> !emitc.opaque<"FHEdouble">
    %c0 = arith.constant 0 : index
    %c20 = arith.constant 20 : index
    %c20_0 = arith.constant 20 : index
    %6 = scf.for %arg4 = %c0 to %c20 step %c20_0 iter_args(%arg5 = %5) -> (!emitc.opaque<"FHEdouble">) {
      %8 = "emitc.constant"() <{value = #emitc.opaque<"std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}">}> : () -> !emitc.opaque<"std::vector<double>">
      %9 = emitc.call "FHEencrypt"(%arg0, %8) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"std::vector<double>">) -> !emitc.opaque<"FHEdouble">
      %10 = "emitc.constant"() <{value = #emitc.opaque<"std::vector<double>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}">}> : () -> !emitc.opaque<"std::vector<double>">
      %11 = emitc.call "FHEencrypt"(%arg0, %10) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"std::vector<double>">) -> !emitc.opaque<"FHEdouble">
      %12 = "emitc.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      %13 = emitc.call "FHEencrypt"(%arg0, %12) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, f64) -> !emitc.opaque<"FHEdouble">
      %14 = "emitc.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      %15 = emitc.call "FHEencrypt"(%arg0, %14) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, f64) -> !emitc.opaque<"FHEdouble">
      %16 = emitc.call "FHEsubf"(%arg0, %arg1, %arg2) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      %17 = emitc.call "FHEmulf"(%arg0, %16, %arg3) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      %18 = emitc.call "FHEeq"(%arg0, %17, %arg3) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      %19 = emitc.call "FHEselect"(%arg0, %18, %11, %9) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      %20 = emitc.call "FHEaddf"(%arg0, %arg5, %19) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"FHEdouble">, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
      scf.yield %20 : !emitc.opaque<"FHEdouble">
    }
    %7 = emitc.call "FHEvectorSum"(%arg0, %6) : (!emitc.ptr<!emitc.opaque<"FHEcontext">>, !emitc.opaque<"FHEdouble">) -> !emitc.opaque<"FHEdouble">
    return %7 : !emitc.opaque<"FHEdouble">
  }
}
