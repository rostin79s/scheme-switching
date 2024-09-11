; ModuleID = 'FHE_module'
source_filename = "FHE_module"

%FHEdouble.5 = type opaque

define %FHEdouble.5 @_Z6rostinii(%FHEdouble.5 %_tmp0, %FHEdouble.5 %_tmp1) {
entry:
  %_tmp3 = call %FHEdouble.5 @FHEmul(%FHEdouble.5 %_tmp0, %FHEdouble.5 %_tmp0)
  %_tmp4 = call %FHEdouble.5 @FHEaddP(%FHEdouble.5 %_tmp3, i32 -2)
  ret %FHEdouble.5 %_tmp4
}

declare %FHEdouble.5 @FHEmul(%FHEdouble.5, %FHEdouble.5)

declare %FHEdouble.5 @FHEaddP(%FHEdouble.5, i32)
