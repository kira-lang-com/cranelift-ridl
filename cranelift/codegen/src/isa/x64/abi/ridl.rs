//! Argument and result assignment for the RIDL calling convention on x86_64
//! (KLF-RIDL specification, section 22).
//!
//! | Base | Integer arguments | Float arguments | Direct results | Indirect result | Context | Error |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | System V | rdi rsi rdx rcx r8 r9 | xmm0 to xmm7 | rax rdx rcx r8, xmm0 to xmm3 | rax | r13 | r12 |
//! | Windows | rcx rdx r8 r9 (positional with xmm0 to xmm3) | xmm0 to xmm3 | rax rdx rcx r8, xmm0 to xmm3 | rax | r13 | r12 |
//!
//! Everything else follows the base standard: stack arguments, shadow space,
//! register preservation.

use crate::ir::{self, ArgumentPurpose, types};
use crate::isa::RidlBase;
use crate::isa::x64::inst::*;
use crate::machinst::*;
use crate::{CodegenError, CodegenResult};
use alloc::borrow::ToOwned;
use smallvec::SmallVec;

/// The indirect result register.
pub(crate) fn indirect_result() -> Reg {
    regs::rax()
}

/// The context register. Callee-saved.
pub(crate) fn context() -> Reg {
    regs::r13()
}

/// The error register. Not preserved.
pub(crate) fn error() -> Reg {
    regs::r12()
}

const RESULT_LEAVES: usize = 4;

fn unsupported(message: &str) -> CodegenError {
    CodegenError::Unsupported(message.to_owned())
}

fn fixed(reg: Reg, param: &ir::AbiParam) -> ABIArg {
    ABIArg::reg(
        reg.to_real_reg().unwrap(),
        param.value_type,
        param.extension,
        param.purpose,
    )
}

fn int_argument(windows: bool, index: usize) -> Option<Reg> {
    let registers: &[fn() -> Reg] = if windows {
        &[regs::rcx, regs::rdx, regs::r8, regs::r9]
    } else {
        &[
            regs::rdi,
            regs::rsi,
            regs::rdx,
            regs::rcx,
            regs::r8,
            regs::r9,
        ]
    };
    registers.get(index).map(|register| register())
}

fn float_argument(windows: bool, index: usize) -> Option<Reg> {
    let registers: &[fn() -> Reg] = &[
        regs::xmm0,
        regs::xmm1,
        regs::xmm2,
        regs::xmm3,
        regs::xmm4,
        regs::xmm5,
        regs::xmm6,
        regs::xmm7,
    ];
    let count = if windows { 4 } else { 8 };
    registers[..count].get(index).map(|register| register())
}

fn int_result(index: usize) -> Option<Reg> {
    let registers: &[fn() -> Reg] = &[regs::rax, regs::rdx, regs::rcx, regs::r8];
    registers.get(index).map(|register| register())
}

fn float_result(index: usize) -> Option<Reg> {
    let registers: &[fn() -> Reg] = &[regs::xmm0, regs::xmm1, regs::xmm2, regs::xmm3];
    registers.get(index).map(|register| register())
}

/// The scalar leaves of a RIDL value, as Cranelift sees them.
fn is_ridl_scalar(ty: ir::Type) -> bool {
    matches!(
        ty,
        types::I8 | types::I16 | types::I32 | types::I64 | types::I128 | types::F32 | types::F64
    )
}

pub(crate) fn compute_arg_locs(
    base: RidlBase,
    params: &[ir::AbiParam],
    args_or_rets: ArgsOrRets,
    add_ret_area_ptr: bool,
    mut args: ArgsAccumulator,
) -> CodegenResult<(u32, Option<usize>)> {
    if add_ret_area_ptr {
        return Err(unsupported(
            "a RIDL result that does not fit in registers returns through an indirect result \
             parameter (`sret`)",
        ));
    }
    let windows = match base {
        RidlBase::SystemV => false,
        RidlBase::WindowsFastcall => true,
        RidlBase::AppleAarch64 => {
            return Err(unsupported("`ridl_apple_aarch64` is an aarch64 convention"));
        }
    };
    let rets = args_or_rets == ArgsOrRets::Rets;
    let mut next_gpr = 0;
    let mut next_xmm = 0;
    // Windows assigns registers by position: each leaf takes the next one.
    let mut position = 0;
    let mut leaves = 0;
    // Windows reserves 32 bytes of shadow space for the four register
    // arguments.
    let mut next_stack: u32 = if windows && !rets { 32 } else { 0 };

    for param in params {
        match param.purpose {
            ArgumentPurpose::StructReturn => {
                args.push(fixed(indirect_result(), param));
                continue;
            }
            ArgumentPurpose::Context if !rets => {
                args.push(fixed(context(), param));
                continue;
            }
            ArgumentPurpose::Error => {
                args.push(fixed(error(), param));
                continue;
            }
            ArgumentPurpose::Context => {
                return Err(unsupported("a RIDL `context` value is never a result"));
            }
            ArgumentPurpose::StructArgument(_) => {
                return Err(unsupported(
                    "RIDL passes aggregates as scalar leaves or by address, not as `sarg`",
                ));
            }
            ArgumentPurpose::Normal | ArgumentPurpose::VMContext => {}
        }
        if !is_ridl_scalar(param.value_type) {
            return Err(unsupported(
                "RIDL leaves are integers of up to 128 bits, `f32` and `f64`",
            ));
        }

        let (classes, parts) = Inst::rc_for_type(param.value_type)?;
        let float = classes[0] == RegClass::Float;
        let registers = (0..parts.len())
            .map(|offset| match (rets, float) {
                (true, false) => int_result(next_gpr + offset),
                (true, true) => float_result(next_xmm + offset),
                (false, false) if windows => int_argument(true, position + offset),
                (false, true) if windows => float_argument(true, position + offset),
                (false, false) => int_argument(false, next_gpr + offset),
                (false, true) => float_argument(false, next_xmm + offset),
            })
            .collect::<Option<SmallVec<[Reg; 2]>>>()
            .filter(|_| !rets || leaves + parts.len() <= RESULT_LEAVES);
        position += parts.len();

        if let Some(registers) = registers {
            let slots = registers
                .iter()
                .zip(parts)
                .map(|(reg, ty)| ABIArgSlot::Reg {
                    reg: reg.to_real_reg().unwrap(),
                    ty: *ty,
                    extension: param.extension,
                })
                .collect();
            args.push(ABIArg::Slots {
                slots,
                purpose: param.purpose,
            });
            if float {
                next_xmm += parts.len();
            } else {
                next_gpr += parts.len();
            }
            leaves += parts.len();
            continue;
        }
        if rets {
            return Err(unsupported(
                "a RIDL result has at most four leaves in registers; return a larger one \
                 through an indirect result parameter (`sret`)",
            ));
        }

        // Every leaf takes an eight-byte slot. System V aligns a 16-byte
        // integer to 16; a value never straddles registers and the stack.
        let size = param.value_type.bytes().max(8);
        if !windows {
            next_stack = align_to(next_stack, size);
        }
        let slots = parts
            .iter()
            .map(|ty| {
                let slot = ABIArgSlot::Stack {
                    offset: i64::from(next_stack),
                    ty: *ty,
                    extension: param.extension,
                };
                next_stack += 8;
                slot
            })
            .collect();
        args.push(ABIArg::Slots {
            slots,
            purpose: param.purpose,
        });
    }

    Ok((align_to(next_stack, 16), None))
}
