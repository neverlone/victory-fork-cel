//! A first-lane read may use either half of a split wave when every active lane
//! holds the same bits and a half with no active lanes cannot observe the result.

use std::collections::HashSet;

use super::Parts;
use crate::gfx10::{Fields, Operand};
use crate::ir::{Reg, Value};
use crate::native::StageIo;
use crate::structure::{Atom, Bool, LaneExit};

pub(super) fn can_read_locally(parts: &Parts<'_>, block: usize, index: usize) -> bool {
    let Some(guard) = parts.st.guards[block][index] else {
        return false;
    };
    let inst = &parts.p.blocks[block].insts[index];
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    if v.dpp.is_some() || v.sdwa.is_some() {
        return false;
    }
    let Some(Operand::Vector { reg, dwords: 1 }) = v.src.first() else {
        return false;
    };
    let Some(value) = inst.read(Reg::V(*reg)) else {
        return false;
    };
    let [(Reg::S(0..=105), result)] = inst.writes.as_slice() else {
        return false;
    };

    uniform_source(parts, block, index, guard, value, &mut HashSet::new())
        && has_only_masked_uses(parts, block, index, guard, *result)
}

/// A lane-local value built in this block from dispatch inputs. A guarded
/// definition must run in every lane the read can select.
fn uniform_source(
    parts: &Parts<'_>,
    block: usize,
    before: usize,
    guard: Bool,
    value: Value,
    proved: &mut HashSet<Value>,
) -> bool {
    if proved.contains(&value) {
        return true;
    }
    let Some((index, writer)) = parts.p.blocks[block].insts[..before]
        .iter()
        .enumerate()
        .rev()
        .find(|(_, inst)| inst.writes.iter().any(|&(_, v)| v == value))
    else {
        return false;
    };
    let Fields::Valu(v) = &writer.fields else {
        return false;
    };
    if !matches!(
        writer.name(),
        "v_mov_b32" | "v_cvt_f32_u32" | "v_rcp_iflag_f32" | "v_mul_legacy_f32" | "v_cvt_u32_f32"
    ) || parts.st.guards[block][index].is_some_and(|g| g != guard)
        || v.dpp.is_some()
        || v.sdwa.is_some()
    {
        return false;
    }
    let StageIo::Cs(cs) = parts.native.io else {
        return false;
    };
    let uniform = v.src.iter().all(|op| match *op {
        Operand::Vector { reg, dwords: 1 } => writer
            .read(Reg::V(reg))
            .is_some_and(|v| uniform_source(parts, block, index, guard, v, proved)),
        Operand::Scalar { code, dwords: 1 } if code < cs.user_sgprs() => writer
            .read(Reg::S(code))
            .is_some_and(|v| parts.p.inputs.contains(&(Reg::S(code), v))),
        Operand::Const(code) => crate::resource::inline(code).is_some(),
        Operand::Literal(_) => true,
        _ => false,
    });
    if uniform {
        proved.insert(value);
    }
    uniform
}

/// The result may pass through scalar arithmetic before a predicated vector
/// instruction consumes it. Outside the mask a host half can have no lane to
/// read. Its arbitrary scalar bits must not reach control flow, a join or an
/// instruction under a different mask.
fn has_only_masked_uses(
    parts: &Parts<'_>,
    block: usize,
    index: usize,
    guard: Bool,
    result: Value,
) -> bool {
    let mut derived = HashSet::from([result]);
    let mut consumed = false;
    let instructions = &parts.p.blocks[block].insts;
    for (i, inst) in instructions.iter().enumerate().skip(index + 1) {
        if !inst.reads.iter().any(|&(_, v)| derived.contains(&v)) {
            continue;
        }
        if matches!(
            inst.name(),
            "s_mov_b32" | "s_add_i32" | "s_mul_i32" | "s_mul_hi_u32"
        ) && inst
            .writes
            .iter()
            .all(|&(r, _)| matches!(r, Reg::S(0..=105) | Reg::Scc))
        {
            derived.extend(inst.writes.iter().map(|&(_, v)| v));
        } else if parts.st.guards[block][i] == Some(guard)
            && matches!(inst.name(), "v_mov_b32" | "v_mul_hi_u32")
            && matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_none() && v.sdwa.is_none())
            && inst.writes.iter().all(|&(r, _)| matches!(r, Reg::V(_)))
        {
            consumed = true;
        } else {
            return false;
        }
    }
    let controls_read_derived = |b| {
        parts.st.bools.support(b).iter().any(|atom| match *atom {
            Atom::Lane(v) | Atom::Uniform(v) | Atom::Phi(v) | Atom::Quad(v) | Atom::Opaque(v) => {
                derived.contains(&v)
            }
            Atom::Constant(_) => false,
        })
    };
    consumed
        && parts.p.blocks.iter().enumerate().all(|(b, data)| {
            !data.phis.iter().any(|phi| phi.args.iter().any(|v| derived.contains(v)))
                && (b == block || !data.insts.iter().any(|inst| inst.reads.iter().any(|&(_, v)| derived.contains(&v))))
                && !matches!(parts.st.exits[b], LaneExit::Branch { when, .. } if controls_read_derived(when))
                && !parts.st.guards[b].iter().flatten().copied().any(controls_read_derived)
        })
}
