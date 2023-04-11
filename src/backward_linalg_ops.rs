use crate::traits::{ArithmeticOps, Dot, HasGrad, Reduce};
use crate::variable::Var;
use std::fmt::Debug;

pub fn dot_backward<'a, T>(parent: &Var<T>, parent_grad: T)
where
    T: HasGrad<T> + ArithmeticOps + Dot<Output = T> + Reduce + Debug,
{
    let l_dep = &parent.deps[0];
    let r_dep = &parent.deps[1];

    if l_dep.requires_grad {
        let l_data = l_dep.data().unwrap();
        let l_current_grad = l_dep.grad().unwrap_or(l_data.get_zero_grad());
        let new_grad = l_current_grad + parent_grad.clone();
        l_dep.set_grad(new_grad);
    }

    if r_dep.requires_grad {
        let r_data = r_dep.data().unwrap();
        let r_current_grad = r_dep.grad().unwrap_or(r_data.get_zero_grad());
        let new_grad = r_current_grad + parent_grad;
        r_dep.set_grad(new_grad);
    }
}
