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

; R's LDOUBLE loops (sum(), mean(), var()): accumulated on the x87 unit with
; 64-bit precision, as R's gcc build does, then rounded to double once.
; Home space: [rsp+8] temporary, [rsp+16] saved / [rsp+18] new control word,
; [rsp+24] n as a double.

SET_PRECISION MACRO
    fnstcw  WORD PTR [rsp+16]
    mov     ax, WORD PTR [rsp+16]
    and     ax, 0F0FFh
    or      ax, 0300h
    mov     WORD PTR [rsp+18], ax
    fldcw   WORD PTR [rsp+18]
ENDM

; double lsm_x87_sum(const double* x, int64 n): sum of x[0..n)
lsm_x87_sum PROC
    SET_PRECISION
    fldz
    test    rdx, rdx
    jle     sum_done
sum_loop:
    fadd    REAL8 PTR [rcx]
    add     rcx, 8
    dec     rdx
    jnz     sum_loop
sum_done:
    fstp    REAL8 PTR [rsp+8]
    fldcw   WORD PTR [rsp+16]
    movsd   xmm0, REAL8 PTR [rsp+8]
    ret
lsm_x87_sum ENDP

; void lsm_x87_accumulate(const double* x, int64 n, void* acc): *acc (an
; 80-bit value) += x[0] + ... + x[n-1], in order
lsm_x87_accumulate PROC
    SET_PRECISION
    fld     TBYTE PTR [r8]
    test    rdx, rdx
    jle     acc_done
acc_loop:
    fadd    REAL8 PTR [rcx]
    add     rcx, 8
    dec     rdx
    jnz     acc_loop
acc_done:
    fstp    TBYTE PTR [r8]
    fldcw   WORD PTR [rsp+16]
    ret
lsm_x87_accumulate ENDP

; double lsm_x87_to_double(const void* acc): an 80-bit value rounded to double
lsm_x87_to_double PROC
    SET_PRECISION
    fld     TBYTE PTR [rcx]
    fstp    REAL8 PTR [rsp+8]
    fldcw   WORD PTR [rsp+16]
    movsd   xmm0, REAL8 PTR [rsp+8]
    ret
lsm_x87_to_double ENDP

; double lsm_x87_mean(const double* x, int64 n), n > 0: R's mean(): s = sum / n,
; then, if (double) s is finite, s += sum(x - s) / n
lsm_x87_mean PROC
    SET_PRECISION
    cvtsi2sd xmm1, rdx
    movsd   REAL8 PTR [rsp+24], xmm1
    mov     r8, rcx
    mov     r9, rdx
    fldz
mean_loop1:
    fadd    REAL8 PTR [rcx]
    add     rcx, 8
    dec     rdx
    jnz     mean_loop1
    fdiv    REAL8 PTR [rsp+24]
    fst     REAL8 PTR [rsp+8]
    mov     rax, QWORD PTR [rsp+8]
    shr     rax, 52
    and     eax, 7FFh
    cmp     eax, 7FFh
    je      mean_done
    fldz
    mov     rcx, r8
    mov     rdx, r9
mean_loop2:
    fld     REAL8 PTR [rcx]
    fsub    st(0), st(2)
    faddp   st(1), st(0)
    add     rcx, 8
    dec     rdx
    jnz     mean_loop2
    fdiv    REAL8 PTR [rsp+24]
    faddp   st(1), st(0)
mean_done:
    fstp    REAL8 PTR [rsp+8]
    fldcw   WORD PTR [rsp+16]
    movsd   xmm0, REAL8 PTR [rsp+8]
    ret
lsm_x87_mean ENDP

; double lsm_x87_ssd(const double* x, int64 n, double mean, int64 divisor):
; sum((x - mean)^2) / divisor, rounded to double once
lsm_x87_ssd PROC
    SET_PRECISION
    cvtsi2sd xmm1, r9
    movsd   REAL8 PTR [rsp+24], xmm1
    movsd   REAL8 PTR [rsp+8], xmm2
    fld     REAL8 PTR [rsp+8]
    fldz
    test    rdx, rdx
    jle     ssd_done
ssd_loop:
    fld     REAL8 PTR [rcx]
    fsub    st(0), st(2)
    fmul    st(0), st(0)
    faddp   st(1), st(0)
    add     rcx, 8
    dec     rdx
    jnz     ssd_loop
ssd_done:
    fdiv    REAL8 PTR [rsp+24]
    fstp    REAL8 PTR [rsp+8]
    fstp    st(0)
    fldcw   WORD PTR [rsp+16]
    movsd   xmm0, REAL8 PTR [rsp+8]
    ret
lsm_x87_ssd ENDP

END
