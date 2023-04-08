use std::fmt::Debug;

use crate::{
    traits::{ArithmeticOps, HasGrad},
    variable::Var,
};

pub fn add_backward<'a, T: HasGrad<T> + ArithmeticOps + Debug>(parent: &Var<T>, parent_grad: T) {
    let l_dep = &parent.deps()[0];
    let r_dep = &parent.deps()[1];

    if l_dep.requires_grad {
        let l_data = l_dep.data().unwrap();
        let l_current_grad = l_dep.grad_or(l_data.get_zero_grad());
        let new_grad = l_current_grad + parent_grad;
        l_dep.set_grad(new_grad);
    }

    if r_dep.requires_grad {
        let r_data = r_dep.data().unwrap();
        let r_current_grad = r_dep.grad_or(r_data.get_zero_grad());
        let new_grad = r_current_grad + parent_grad;
        r_dep.set_grad(new_grad);
    }
}

pub fn mul_backward<'a, T: HasGrad<T> + ArithmeticOps + Debug>(parent: &Var<T>, parent_grad: T) {
    let l_dep = &parent.deps[0];
    let r_dep = &parent.deps[1];

    if l_dep.requires_grad {
        let l_data = l_dep.data().unwrap();
        let r_data = r_dep.data().unwrap();

        let l_current_grad = l_dep.grad_or(l_data.get_zero_grad());
        let new_grad = l_current_grad + parent_grad * r_data;
        l_dep.set_grad(new_grad);
    }

    if r_dep.requires_grad {
        let r_data = r_dep.data().unwrap();
        let l_data = l_dep.data().unwrap();

        let r_current_grad = r_dep.grad_or(r_data.get_zero_grad());
        let new_grad = r_current_grad + parent_grad * l_data;
        r_dep.set_grad(new_grad);
    }
}
