//! Argument and result assignment for the RIDL calling convention on aarch64
//! (KLF-RIDL specification, section 22).
//!
//! | Integer arguments | Float arguments | Direct results | Indirect result | Context | Error |
//! | --- | --- | --- | --- | --- | --- |
//! | x0 to x7 | v0 to v7 | x0 to x3, v0 to v3 | x8 | x20 | x21 |
//!
//! Everything else follows the base standard: AAPCS64, or its Apple variant
//! for stack arguments.

use crate::ir::{self, ArgumentPurpose, types};
use crate::isa::RidlBase;
use crate::isa::aarch64::inst::*;
use crate::machinst::*;
use crate::{CodegenError, CodegenResult};
use alloc::borrow::ToOwned;

/// The indirect result register.
pub(crate) const INDIRECT_RESULT: u8 = 8;
/// The context register. Callee-saved.
pub(crate) const CONTEXT: u8 = 20;
/// The error register. Not preserved.
pub(crate) const ERROR: u8 = 21;

const ARGUMENT_REGISTERS: u8 = 8;
const RESULT_REGISTERS: u8 = 4;
const RESULT_LEAVES: u8 = 4;

fn unsupported(message: &str) -> CodegenError {
    CodegenError::Unsupported(message.to_owned())
}

fn fixed(register: u8, param: &ir::AbiParam) -> ABIArg {
    ABIArg::reg(
        xreg(register).to_real_reg().unwrap(),
        param.value_type,
        param.extension,
        param.purpose,
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
    let apple = base == RidlBase::AppleAarch64;
    let rets = args_or_rets == ArgsOrRets::Rets;
    let class_registers = if rets {
        RESULT_REGISTERS
    } else {
        ARGUMENT_REGISTERS
    };
    let mut next_xreg = 0;
    let mut next_vreg = 0;
    let mut leaves = 0;
    let mut next_stack: u32 = 0;

    for param in params {
        match param.purpose {
            ArgumentPurpose::StructReturn => {
                args.push(fixed(INDIRECT_RESULT, param));
                continue;
            }
            ArgumentPurpose::Context if !rets => {
                args.push(fixed(CONTEXT, param));
                continue;
            }
            ArgumentPurpose::Error => {
                args.push(fixed(ERROR, param));
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

        if !matches!(
            param.value_type,
            types::I8 | types::I16 | types::I32 | types::I64 | types::I128 | types::F32 | types::F64
        ) {
            return Err(unsupported(
                "RIDL leaves are integers of up to 128 bits, `f32` and `f64`",
            ));
        }

        let (classes, parts) = Inst::rc_for_type(param.value_type)?;
        let count = u8::try_from(parts.len()).unwrap();
        let next = match classes[0] {
            RegClass::Int => &mut next_xreg,
            RegClass::Float => &mut next_vreg,
            RegClass::Vector => unreachable!(),
        };
        let fits = *next + count <= class_registers && (!rets || leaves + count <= RESULT_LEAVES);
        if fits {
            let slots = classes
                .iter()
                .zip(parts)
                .enumerate()
                .map(|(index, (class, ty))| {
                    let number = *next + u8::try_from(index).unwrap();
                    let reg = match class {
                        RegClass::Int => xreg(number),
                        RegClass::Float => vreg(number),
                        RegClass::Vector => unreachable!(),
                    };
                    ABIArgSlot::Reg {
                        reg: reg.to_real_reg().unwrap(),
                        ty: *ty,
                        extension: param.extension,
                    }
                })
                .collect();
            args.push(ABIArg::Slots {
                slots,
                purpose: param.purpose,
            });
            *next += count;
            leaves += count;
            continue;
        }
        if rets {
            return Err(unsupported(
                "a RIDL result has at most four leaves in registers; return a larger one \
                 through an indirect result parameter (`sret`)",
            ));
        }

        // The class is exhausted for this and every later argument, as in
        // AAPCS64 stage C: a value never straddles registers and the stack.
        *next = class_registers;
        let size = param.value_type.bytes();
        let size = if apple { size } else { size.max(8) };
        debug_assert!(size.is_power_of_two());
        next_stack = align_to(next_stack, size);
        let mut offset = next_stack;
        let slots = parts
            .iter()
            .map(|ty| {
                let slot = ABIArgSlot::Stack {
                    offset: i64::from(offset),
                    ty: *ty,
                    extension: param.extension,
                };
                offset += ty.bytes();
                slot
            })
            .collect();
        args.push(ABIArg::Slots {
            slots,
            purpose: param.purpose,
        });
        next_stack += size;
    }

    Ok((align_to(next_stack, 16), None))
}
