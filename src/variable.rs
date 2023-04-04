use std::fmt::Debug;

use uuid::Uuid;

use crate::context::{ArithmeticOps, Context};

#[derive(Copy, Clone, Debug)]
pub enum VarType {
    Leaf,
    OpAdd,
    OpSub,
    OpMul,
    OpDiv,
}

pub trait GetZeroGrad<T> {
    /// Get zero gradient value according to the value.
    /// For scalar, normally it returns zero. For ndarray, it will return
    /// ndarray of zeros with the same shape as the data itself.
    fn get_zero_grad(&self) -> T;
}

#[derive(Clone)]
pub struct BackwardFn<T> {
    pub func: fn(context: &mut Context<T>, &mut Variable<T>),
}

// #[derive(Clone)]
// type BackwardFn<T> = fn(context: &mut Context<T>, &mut Variable<T>);
impl<T> Debug for BackwardFn<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<BackwardFn>")
    }
}

fn add_backward<T: ArithmeticOps>(_context: &mut Context<T>, _parent: &mut Variable<T>) {
    todo!()
    // let lbf = move || {
    //     let parent_grad = *self.gradient_map.get(&parent_id.clone()).unwrap();
    //     let mut l_grad = *self.gradient_map.get(&l_id.clone()).unwrap();
    //     let mut r_grad = *self.gradient_map.get(&rdep.data_id).unwrap();

    //     l_grad = l_grad + parent_grad;
    //     self.gradient_map.insert(ldep.data_id, l_grad);

    //     r_grad = r_grad + parent_grad;
    //     self.gradient_map.insert(rdep.data_id, r_grad);
    // };
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
                func: add_backward::<T>,
            }),
        )
    }

    pub fn sub(&self, other: &Variable<T>) -> Variable<T> {
        self.make_binop(other, VarType::OpSub, None)
    }

    pub fn mul(&self, other: &Variable<T>) -> Variable<T> {
        self.make_binop(other, VarType::OpMul, None)
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

impl<T: ArithmeticOps> std::ops::Add<&Variable<T>> for &Variable<T> {
    type Output = Variable<T>;

    fn add(self, rhs: &Variable<T>) -> Self::Output {
        self.add(rhs)
    }
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
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
