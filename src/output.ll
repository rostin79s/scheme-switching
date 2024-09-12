; ModuleID = 'FHE_module'
source_filename = "FHE_module"

%FHEdouble = type opaque

define %FHEdouble @rostin(%FHEdouble %_tmp0, %FHEdouble %_tmp1) {
entry:
  %_tmp3 = call %FHEdouble @FHEmul(%FHEdouble %_tmp0, %FHEdouble %_tmp0)
  %_tmp4 = call %FHEdouble @FHEsubP(i32 -2, %FHEdouble %_tmp0)
  %_tmp5 = call %FHEdouble @FHEadd(%FHEdouble %_tmp4, %FHEdouble %_tmp3)
  ret %FHEdouble %_tmp5
}

declare %FHEdouble @FHEmul(%FHEdouble, %FHEdouble)

declare %FHEdouble @FHEsubP(i32, %FHEdouble)

declare %FHEdouble @FHEadd(%FHEdouble, %FHEdouble)

define i32 @main() {
entry:
  ret i32 0
}
