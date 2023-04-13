use std::fmt::Debug;

use crate::traits::{ArithmeticOps, Dot, HasGrad, Reduce, Shape};
use crate::variable::Var;

fn compute_broadcasted_gradients<'a, T>(data: &T, parent_grad: &T) -> T
where
    T: HasGrad<T> + ArithmeticOps + Dot<Output = T> + Reduce + Shape + Debug,
{
    let mut parent_grad = parent_grad.clone();
    // Sum out added dims
    let ndims_added = parent_grad.ndim() - data.ndim();
    for _ in 0..ndims_added {
        parent_grad = parent_grad.sum_axis(ndims_added);
    }

    // Sum across broadcasted (but non-added dims)
    for i in 0..data.ndim() {
        if data.shape()[i] == 1 {
            parent_grad = parent_grad.sum_axis(i);
        }
    }

    parent_grad
}

pub fn add_backward<'a, T>(parent: &Var<T>, parent_grad: T)
where
    T: HasGrad<T> + ArithmeticOps + Shape + Dot<Output = T> + Reduce + Debug,
{
    let l_dep = &parent.deps[0];
    let r_dep = &parent.deps[1];

    if l_dep.requires_grad {
        let l_data = l_dep.data().unwrap();
        let parent_grad = compute_broadcasted_gradients(&l_data, &parent_grad);

        let l_current_grad = l_dep.grad().unwrap_or(l_data.get_zero_grad());
        let new_grad = l_current_grad + parent_grad.clone();
        l_dep.set_grad(new_grad);
    }

    if r_dep.requires_grad {
        let r_data = r_dep.data().unwrap();
        let parent_grad = compute_broadcasted_gradients(&r_data, &parent_grad);

        let r_current_grad = r_dep.grad().unwrap_or(r_data.get_zero_grad());
        let new_grad = r_current_grad + parent_grad;
        r_dep.set_grad(new_grad);
    }
}

pub fn sub_backward<'a, T>(parent: &Var<T>, parent_grad: T)
where
    T: HasGrad<T> + ArithmeticOps + Dot<Output = T> + Reduce + Debug,
{
    let l_dep = &parent.deps()[0];
    let r_dep = &parent.deps()[1];

    if l_dep.requires_grad {
        let l_data = l_dep.data().unwrap();
        let l_current_grad = l_dep.grad().unwrap_or(l_data.get_zero_grad());
        let new_grad = l_current_grad + parent_grad.clone();
        l_dep.set_grad(new_grad);
    }

    if r_dep.requires_grad {
        let r_data = r_dep.data().unwrap();
        let r_current_grad = r_dep.grad().unwrap_or(r_data.get_zero_grad());
        let new_grad = r_current_grad - parent_grad; // Note the subtraction here
        r_dep.set_grad(new_grad);
    }
}

pub fn mul_backward<'a, T>(parent: &Var<T>, parent_grad: T)
where
    T: HasGrad<T> + ArithmeticOps + Dot<Output = T> + Reduce + Debug,
{
    let l_dep = &parent.deps[0];
    let r_dep = &parent.deps[1];

    if l_dep.requires_grad {
        let l_data = l_dep.data().unwrap();
        let r_data = r_dep.data().unwrap();

        let l_current_grad = l_dep.grad().unwrap_or(l_data.get_zero_grad());
        let new_grad = l_current_grad + parent_grad.clone() * r_data;
        l_dep.set_grad(new_grad);
    }

    if r_dep.requires_grad {
        let r_data = r_dep.data().unwrap();
        let l_data = l_dep.data().unwrap();

        let r_current_grad = r_dep.grad().unwrap_or(r_data.get_zero_grad());
        let new_grad = r_current_grad + parent_grad * l_data;
        r_dep.set_grad(new_grad);
    }
}
