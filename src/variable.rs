use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

use uuid::Uuid;

use crate::{
    backward_funcs::{self, BackwardFn},
    traits::ArithmeticOps,
};

#[derive(Copy, Clone, Debug)]
pub enum VarType {
    Leaf,
    OpAdd,
    OpSub,
    OpMul,
    OpDiv,
}

#[derive(Debug)]
pub struct Variable<T> {
    pub backward_fn: Option<BackwardFn<T>>,
    pub data_id: Uuid,
    pub deps: Vec<Box<Variable<T>>>,
    pub is_leaf: bool,
    pub label: String,
    pub requires_grad: bool,
    pub var_type: VarType,
}

impl<T: ArithmeticOps> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Self {
            backward_fn: self.backward_fn.clone(),
            data_id: self.data_id,
            deps: self.deps.clone(),
            is_leaf: self.is_leaf,
            label: self.label.clone(),
            requires_grad: self.requires_grad,
            var_type: self.var_type.clone(),
        }
    }
}

impl<'a, T: ArithmeticOps> Variable<T> {
    fn make_binop(
        &self,
        other: &Variable<T>,
        op: VarType,
        backward_func: Option<BackwardFn<T>>,
    ) -> Variable<T> {
        let mut new_var = Variable::default();
        new_var.var_type = op;
        new_var.deps.push(Box::new(self.clone()));
        new_var.deps.push(Box::new(other.clone()));
        new_var.requires_grad = self.requires_grad || other.requires_grad;
        new_var.is_leaf = false;
        new_var.backward_fn = backward_func;

        new_var
    }

    pub fn add(&self, other: &Variable<T>) -> Variable<T> {
        self.make_binop(
            other,
            VarType::OpAdd,
            Some(BackwardFn {
                func: backward_funcs::add_backward::<T>,
                name: "Add".to_string(),
            }),
        )
    }

    pub fn sub(&self, other: &Variable<T>) -> Variable<T> {
        self.make_binop(other, VarType::OpSub, None)
    }

    pub fn mul(&self, other: &Variable<T>) -> Variable<T> {
        self.make_binop(
            other,
            VarType::OpMul,
            Some(BackwardFn {
                func: backward_funcs::mul_backward::<T>,
                name: "Mul".to_string(),
            }),
        )
    }

    pub fn div(&self, other: &Variable<T>) -> Variable<T> {
        self.make_binop(other, VarType::OpDiv, None)
    }

    pub fn set_label(&mut self, label: String) {
        self.label = label;
    }

    pub fn data_id(&self) -> Uuid {
        self.data_id
    }
}

impl<'a, T: ArithmeticOps> Default for Variable<T> {
    fn default() -> Self {
        Self {
            backward_fn: None,
            data_id: Uuid::new_v4(),
            deps: Default::default(),
            is_leaf: true,
            label: Default::default(),
            requires_grad: false,
            var_type: VarType::Leaf,
        }
    }
}

impl<T: ArithmeticOps> Add<&Variable<T>> for &Variable<T> {
    type Output = Variable<T>;

    fn add(self, rhs: &Variable<T>) -> Self::Output {
        self.add(rhs)
    }
}

impl<T: ArithmeticOps> Mul<&Variable<T>> for &Variable<T> {
    type Output = Variable<T>;

    fn mul(self, rhs: &Variable<T>) -> Self::Output {
        self.mul(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;

    #[test]
    fn test_creation() {
        let mut a: Variable<i32> = Default::default();
        let mut b: Variable<i32> = Default::default();
        a.set_label("a".to_string());
        b.set_label("b".to_string());

        let res = &a.add(&b);
        assert!(res.deps.len() == 2);

        let res = &a.add(&a);
        assert!(res.deps[0].data_id == res.deps[1].data_id);

        let mut c = Context::new();
        let x = &c.var(1);
        let y = &c.var(2);
        let z = &c.var(3);

        let result = (x + y).add(z);
        assert!(c.value_of(&result) == 6);
    }
}
