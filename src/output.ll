; ModuleID = 'FHE_module'
source_filename = "FHE_module"

%FHEdouble = type opaque

define %FHEdouble @rostin(%FHEdouble %_tmp0, %FHEdouble %_tmp1) {
entry:
  %_tmp3 = call %FHEdouble @FHEmul(%FHEdouble %_tmp1, %FHEdouble %_tmp0)
  %_tmp4 = call %FHEdouble @FHEmul(%FHEdouble %_tmp3, %FHEdouble %_tmp1)
  ret %FHEdouble %_tmp4
}

declare %FHEdouble @FHEmul(%FHEdouble, %FHEdouble)

define i32 @main() {
entry:
  ret i32 0
}
