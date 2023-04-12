use crate::traits::{ArithmeticOps, Dot, HasGrad, Reduce, Transpose};
use crate::variable::Var;
use std::fmt::Debug;


pub fn dot_backward<'a, T>(parent: &Var<T>, parent_grad: T) 
where
    T: HasGrad<T> + ArithmeticOps + Dot<Output = T> +Transpose+ Reduce + Debug,
{
    let l_dep = &parent.deps[0];
    let r_dep = &parent.deps[1];
    
    if l_dep.requires_grad {
        let grad_wrt_left = parent_grad.dot(r_dep.data().unwrap().t());
        let current_grad = l_dep.grad().unwrap_or(r_dep.data().unwrap().get_zero_grad());
        let new_grad = current_grad + grad_wrt_left;
        parent.deps[0].set_grad(new_grad);
    }

    if r_dep.requires_grad {
        let grad_wrt_right = l_dep.data().unwrap().t().dot(parent_grad);
        let current_grad = r_dep.grad().unwrap_or(r_dep.data().unwrap().get_zero_grad());
        let new_grad = current_grad + grad_wrt_right;
        parent.deps[1].set_grad(new_grad);
    }

}
