use std::fmt::Debug;

use crate::{context::Context, traits::ArithmeticOps, variable::Variable};

#[derive(Clone)]
pub struct BackwardFn<T> {
    pub func: fn(context: &mut Context<T>, &Variable<T>),
    pub name: String,
}

impl<T> Debug for BackwardFn<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<BackwardFn: {}>", self.name)
    }
}

pub fn add_backward<T: ArithmeticOps>(context: &mut Context<T>, parent: &Variable<T>) {
    let parent_grad = *context.gradient_map.get(&parent.data_id).unwrap();

    let l_dep = &parent.deps[0];
    let r_dep = &parent.deps[1];

    if l_dep.requires_grad {
        let l_id = l_dep.data_id;
        let mut l_grad = *context.gradient_map.get(&l_id).unwrap();
        l_grad = l_grad + parent_grad;
        context.gradient_map.insert(l_dep.data_id, l_grad);
    }

    if r_dep.requires_grad {
        let r_id = r_dep.data_id;
        let mut r_grad = *context.gradient_map.get(&r_id).unwrap();
        r_grad = r_grad + parent_grad;
        context.gradient_map.insert(r_dep.data_id, r_grad);
    }
}

pub fn mul_backward<T: ArithmeticOps>(context: &mut Context<T>, parent: &Variable<T>) {
    let parent_grad = *context.gradient_map.get(&parent.data_id).unwrap();

    let l_dep = &parent.deps[0];
    let r_dep = &parent.deps[1];
    let l_id = l_dep.data_id;
    let r_id = r_dep.data_id;

    if l_dep.requires_grad {
        let r_val = *context.data_map.get(&r_id).unwrap();
        let l_local_grad = parent_grad * r_val;

        let mut l_current_grad = *context.gradient_map.get(&l_id).unwrap();
        l_current_grad = l_current_grad + l_local_grad;
        context.gradient_map.insert(l_id, l_current_grad);
    }

    if r_dep.requires_grad {
        let l_val = *context.data_map.get(&l_id).unwrap();
        let r_local_grad = parent_grad * l_val;

        let mut r_current_grad = *context.gradient_map.get(&r_id).unwrap();
        r_current_grad = r_current_grad + r_local_grad;
        context.gradient_map.insert(r_id, r_current_grad);
    }
}
