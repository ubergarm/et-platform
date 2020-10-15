# ------------------------------------------------------------------------- 
#  Copyright (C) 2019, Esperanto Technologies Inc
#  The copyright to the computer program(s) herein is the
#  property of Esperanto Technologies, Inc. All Rights Reserved. 
#  The program(s) may be used and/or copied only with
#  the written permission of Esperanto Technologies and
#  in accordance with the terms and conditions stipulated in the
#  agreement/contract under which the program(s) have been supplied.
# ------------------------------------------------------------------------- 


.equ FloatTy,   0
.equ Float16Ty, 1
.equ Int8QTy,   2
.equ Int16QTy,  4
.equ Int32QTy,  5

.data
        
.align 8
__offsets8:
.word 0
.word 1
.word 2
.word 3
.word 4
.word 5
.word 6
.word 7

__offsets16:
.word 0
.word 2
.word 4
.word 6
.word 8
.word 10
.word 12
.word 14

.text
        
.macro FROM_INTX_TO_FP32 reg
        #It doesn't care the intX(type)
        #reg is Int8 or Int16 and It will be convert to Float32
        fsub.pi    \reg, \reg, f16  
        fcvt.ps.pw \reg, \reg    
        fmul.ps    \reg,\reg, f17       
.endm

        
.macro FROM_FP32_TO_INTX elk, reg
        #elk defines the output on reg it would be Int8 or Int16
        frcp.ps    f17, f17                         
        fcvt.ps.pw f16, f16                      
        fmadd.ps   \reg,  \reg, f17, f16  
        fcvt.pw.ps \reg, \reg
        .if \elk == Int8QTy
          #dst = src > (2^8-1) ? (2^8-1) : src < (-2^8) ? (-2^8) : src
          fsat8.pi \reg, \reg
        .else  #Assumes elk=Int16QTy
          #dst = src > (2^15-1) ? (2^15-1) : src < (-2^15) ? (-2^15) : src
          fbci.pi f27, (1<<15)-1                     
          fmin.pi \reg, \reg, f27        
          fnot.pi f27, f27                         
          fmax.pi \reg, \reg, f27
        .endif
.endm


.macro ACT_x_WGT_COMMON
        fmul.ps   f0, f0, f1          
        fswizz.ps f1, f0, 0xe         
        fadd.ps   f0, f0, f1          
        fswizz.ps f1, f0, 0x1         
        fadd.ps   f0, f0, f1          
        fadd.ps   f31, f0, f31
.endm

        
.macro ACT_x_WGT_FloatXTy actAddr, wgtAddr, elk
        .if \elk == FloatTy
          flw.ps      f0, 0x0(\actAddr)
          fgw.ps      f1, f29(\wgtAddr) 
        .else #Float16Ty
          fgh.ps      f0, f28(\actAddr)
          fgh.ps      f1, f29(\wgtAddr)
          fcvt.ps.f16 f0, f0
          fcvt.ps.f16 f1, f1
        .endif
        ACT_x_WGT_COMMON
.endm
        
        
.macro ACT_x_WGT_IntXQTy actAddr, wgtAddr, scale, offset, elk
        .if \elk == Int8QTy
          fgb.ps  f0, f28(\actAddr)   
          fgb.ps  f1, f29(\wgtAddr)
        .else #Int16QTy
          fgh.ps  f0, f28(\actAddr)
          fgh.ps  f1, f29(\wgtAddr)       
        .endif
        fbc.ps   f16, 0x0(\offset)  
        fbc.ps   f17, 0x0(\scale)   
        FROM_INTX_TO_FP32 f0                   
        fbc.ps   f16, 0x4(\offset)
        fbc.ps   f17, 0x4(\scale)
        FROM_INTX_TO_FP32 f1                   
        ACT_x_WGT_COMMON
.endm


.macro ACT_x_WGT actAddr, wgtAddr, scale, offset, elk
	.if ((\elk == FloatTy) || (\elk == Float16Ty))
	  ACT_x_WGT_FloatXTy \actAddr, \wgtAddr, \elk
	.else #Int8QTy || Int16QTy
	  ACT_x_WGT_IntXQTy \actAddr, \wgtAddr, \scale, \offset, \elk
	.endif
.endm

	
.macro INIT_STEP gthValuesWgt, wgtRegStep, elemsRow, elk, belk

        mov.m.x m0, zero, 0xff

        #FloatTy doesn't use gather for activation values
       .if \elk == Int8QTy
          la t0, __offsets8
          flw.ps f28, 0(t0)
       .elseif ((\elk == Int16QTy) || (\elk == Float16Ty))
          la t0, __offsets16
          flw.ps f28, 0(t0)
       .endif

        li t0, 0
	
        #flw.ps f29, 0x0(\gthValuesWgt)
        #in cases of \"m\" on call flw.ps f29, \gthValuesWgt
	flw.ps f29, \gthValuesWgt
        fbc.ps  f30, 0x0(\wgtRegStep)
        fxor.pi f31, f31, f31
        .if (((\elk == Int8QTy) && (\belk == Int8QTy)) || ((\elk == Int16QTy) && (\belk == Int16QTy)))
          fxor.pi f27, f27, f27 
        .endif

        1:
        addi    t0, t0, 8
        ble     \elemsRow, t0, 2f
.endm

        
.macro NEXT_CHUNK_STEP actAddr, elemsRow, elk

        .if \elk == FloatTy
          addi \actAddr, \actAddr, 0x20
        .elseif \elk == Int8QTy
          addi \actAddr, \actAddr, 0x08
        .else  #Float16Ty Int16QTy
          addi \actAddr, \actAddr, 0x10
        .endif
        fadd.pi f29, f29, f30
        beq      zero, zero, 1b

        2:
        fxor.pi  f0, f0, f0
        addi     t0, t0, -8
        sub      t0, \elemsRow, t0
        addi     t1, zero, 1
        sll      t1, t1, t0
        addi     t1, t1, -1
        mov.m.x  m0, t1, 0
.endm

        
.macro ADD_BIAS_STEP biasAddr, dstAddr, scale, offset, elk, belk
        #elk denotes the types of dst act wght
        #belk denotes the type of bias it can be different from elk only if elk is 2 or 4
        #elk and blk get these values 0-FloatTy 1-Float16Ty 2-Int8QTy 4-Int16QTy 5-Int32QTy
        fmvs.x.ps t0, f31, 0x4
        fmv.w.x   f30, t0
        fadd.s    f31, f30, f31
        mov.m.x   m0, zero, 0x1

        .if \elk == FloatTy
          fbc.ps    f30, 0x0(\biasAddr)
        .elseif \elk == Float16Ty
          fbc.ps    f30, 0x0(\biasAddr) 
          fcvt.ps.f16 f30, f30
        .elseif \elk == Int8QTy  
          .if \belk == FloatTy
            fbc.ps f30, 0x0(\biasAddr)
          .elseif ((\belk == Int8QTy) || (\belk == Int32QTy))
             .if \belk == Int8QTy
                fgb.ps f30, f27(\biasAddr)
             .else
                fbc.ps f30, 0x0(\biasAddr)
             .endif
             #dequantize  biasAddr uses value at[3]
             fbc.ps f16, 0x8(\offset)
             fbc.ps f17, 0x8(\scale)
             FROM_INTX_TO_FP32 f30
          .endif
        .elseif \elk == Int16QTy
          .if \belk == Int16QTy 
            fgh.ps f30, f27(\biasAddr)  
          .elseif \belk == 5 #Int32QTy
            fbc.ps f30, 0x0(\biasAddr) 
          .endif
          #dequantize biasAddr uses value at[3] 
          fbc.ps f16, 0x8(\offset)   
          fbc.ps f17, 0x8(\scale)
          FROM_INTX_TO_FP32 f30 
        .endif

        fadd.s    f31, f30, f31

        .if \elk == FloatTy
          fsw.ps  f31, 0x0(\dstAddr)
        .elseif \elk == Float16Ty
          fcvt.f16.ps f31, f31
          fsch.ps f31, f28(\dstAddr)
        .elseif \elk == Int8QTy
          #belk is FloatTy or Int8Qty or Int32QTy
          #quantize output uses value at[4]
          fbc.ps f16, 0xc(\offset)
          fbc.ps f17, 0xc(\scale)
          FROM_FP32_TO_INTX \elk, f31
          fscb.ps f31, f28(\dstAddr)
        .elseif \elk == Int16QTy
            #\belk == Int16QTy or Int32QTy
            #quantize output uses value at[4]
            fbc.ps f16, 0xc(\offset)  
            fbc.ps f17, 0xc(\scale) 
            FROM_FP32_TO_INTX \elk, f31
            fsch.ps f31, f28(\dstAddr)
        .endif
.endm
