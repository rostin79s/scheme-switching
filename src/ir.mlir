module {
  func.func @_Z6rostindd(%arg0: f64, %arg1: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = emitc.call "FHEaddf"(%arg0, %arg1) {args = [], template_args = []} : (f64, f64) -> f64
    %1 = emitc.call "FHEaddf"(%0, %arg0) {args = [], template_args = []} : (f64, f64) -> f64
    %2 = emitc.call "FHEsubf"(%1, %arg1) {args = [], template_args = []} : (f64, f64) -> f64
    %3 = emitc.call "FHEmulf"(%2, %2) {args = [], template_args = []} : (f64, f64) -> f64
    %4 = emitc.call "FHEmulf"(%arg1, %3) {args = [], template_args = []} : (f64, f64) -> f64
    return %4 : f64
  }
}
