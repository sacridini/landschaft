; SPDX-License-Identifier: GPL-3.0-or-later
;
; log() and sqrt() as R computes them on x86-64 Windows: R links the
; mingw-w64 math library, whose log (glibc's __ieee754_log) and sqrt run on
; the x87 unit with 64-bit precision and round the result to double once
; more. MSVC has no inline assembly on x64, hence this file. The x87
; control word is set to 64-bit precision, round to nearest, and restored.
; Leaf functions: they only use the caller's home space.

.data
one     REAL8 1.0
limit   REAL8 0.29

.code

; double lsm_x87_log(double x), x > 0 and finite
lsm_x87_log PROC
    movsd   REAL8 PTR [rsp+8], xmm0
    fnstcw  WORD PTR [rsp+16]
    mov     ax, WORD PTR [rsp+16]
    and     ax, 0F0FFh
    or      ax, 0300h
    mov     WORD PTR [rsp+18], ax
    fldcw   WORD PTR [rsp+18]
    fldln2
    fld     REAL8 PTR [rsp+8]
    fld     st(0)
    fsub    one
    fld     st(0)
    fabs
    fcomp   limit
    fnstsw  ax
    and     ah, 45h
    jz      use_fyl2x
    fstp    st(1)
    fyl2xp1
    jmp     done
use_fyl2x:
    fstp    st(0)
    fyl2x
done:
    fstp    REAL8 PTR [rsp+8]
    fldcw   WORD PTR [rsp+16]
    movsd   xmm0, REAL8 PTR [rsp+8]
    ret
lsm_x87_log ENDP

; double lsm_x87_sqrt(double x), x >= 0
lsm_x87_sqrt PROC
    movsd   REAL8 PTR [rsp+8], xmm0
    fnstcw  WORD PTR [rsp+16]
    mov     ax, WORD PTR [rsp+16]
    and     ax, 0F0FFh
    or      ax, 0300h
    mov     WORD PTR [rsp+18], ax
    fldcw   WORD PTR [rsp+18]
    fld     REAL8 PTR [rsp+8]
    fsqrt
    fstp    REAL8 PTR [rsp+8]
    fldcw   WORD PTR [rsp+16]
    movsd   xmm0, REAL8 PTR [rsp+8]
    ret
lsm_x87_sqrt ENDP

END
