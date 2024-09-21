module {
  func.func @_Z6rostindd(%arg0: !emitc.ptr<!emitc.opaque<"FHEdouble">>, %arg1: !emitc.ptr<!emitc.opaque<"FHEdouble">>) -> !emitc.ptr<!emitc.opaque<"FHEdouble">> attributes {llvm.linkage = #llvm.linkage<external>} {
    return %arg0 : !emitc.ptr<!emitc.opaque<"FHEdouble">>
  }
}
