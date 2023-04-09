use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::vec;
use traits_ndarray::NDArray;

use ndarray::{Array, CowArray, Dimension};

use crate::backward_basic_ops::{add_backward, mul_backward, sub_backward};
use crate::traits::{ArithmeticOps, HasGrad};
use crate::traits_ndarray;

#[derive(Clone, Copy, Debug, PartialEq)]
enum VariableType {
    Input,
    OpAdd,
    OpSub,
    OpMul,
}

type BackwardFn<T> = fn(&Var<T>, T);
type Deps<T> = Vec<Box<Var<T>>>;

pub struct Var<T> {
    pub(crate) deps: Deps<T>,
    pub(crate) requires_grad: bool,
    pub(crate) id: uuid::Uuid,
    pub(crate) backward_fn: Option<BackwardFn<T>>,
    pub(crate) data: Rc<RefCell<Option<T>>>,
    pub(crate) grad: Rc<RefCell<Option<T>>>,
    pub(crate) evaluated: bool,
    pub(crate) is_leaf: bool,
    var_type: VariableType,
}

impl<T: Display + Clone> Debug for Var<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.data.borrow().clone() {
            Some(_) => {
                write!(
                    f,
                    "Variable containing:\n{}",
                    self.data.borrow().clone().unwrap()
                )
            }
            None => {
                write!(f, "Variable contains no data, possible because it is an intermediary variable. Invoke .eval() first.",)
            }
        }
    }
}

pub fn from_ndarray<'a, D: Dimension>(data: Array<f64, D>) -> Var<NDArray<'a>> {
    let data = NDArray(CowArray::from(data.into_dyn()));
    Var::new(data)
}

impl<'a, T: HasGrad<T> + ArithmeticOps + Debug> Var<T> {
    pub fn new(data: T) -> Self {
        Var {
            id: uuid::Uuid::new_v4(),
            data: Rc::new(RefCell::new(Some(data))),
            deps: vec![],
            evaluated: false,
            grad: Rc::new(RefCell::new(None)),
            is_leaf: true,
            requires_grad: false,
            var_type: VariableType::Input,
            backward_fn: None,
        }
    }

    fn dfs(
        variable: &Var<T>,
        visited: &mut HashSet<uuid::Uuid>,
        stack: &mut Vec<Var<T>>,
        allow_revisit: bool,
    ) {
        if allow_revisit || !visited.contains(&variable.id) {
            visited.insert(variable.id);
            for dep in &variable.deps {
                Self::dfs(dep, visited, stack, allow_revisit);
            }
            stack.push(variable.copy());
        }
    }

    fn topological_sort(entry: &Var<T>, allow_revisit: bool) -> Vec<Var<T>> {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        Self::dfs(entry, &mut visited, &mut stack, allow_revisit);
        stack.into_iter().collect()
    }

    /// Make a cheap copy of self. Copying will create a new structure, but
    /// the copy will share the same data and grad with the original Var.
    fn copy(&self) -> Self {
        Self {
            id: self.id,
            data: Rc::clone(&self.data),
            deps: self.deps.iter().map(|v| Box::new(v.copy())).collect(),
            evaluated: self.evaluated,
            grad: Rc::clone(&self.grad),
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            var_type: self.var_type,
            backward_fn: self.backward_fn,
        }
    }

    pub fn backward(&mut self) {
        self.eval();

        // backward requires reverse-topological sort, allowing revisiting
        let mut sorted = Self::topological_sort(self, true);
        sorted.reverse();

        // run backward propagation in iterative manner
        for i in 0..sorted.len() {
            let var = &mut sorted[i];
            match var.backward_fn {
                Some(bw_fn) => {
                    // var requires grad. Proceed.
                    let var_val = var.data.borrow().clone().unwrap();

                    // The grad of root node is set from get_default_init_grad(), which is
                    // usually ones. Otherwise, get the grad from the grad_map by that node's id
                    let grad = if var.id == self.id {
                        var_val.get_default_init_grad()
                    } else {
                        var.grad.borrow().clone().unwrap()
                    };

                    bw_fn(var, grad);
                }
                None => (),
            }
        }
    }

    pub fn eval_bin_op(parent: &Var<T>, op: fn(T, T) -> T) {
        let ldata = parent.deps[0].data.borrow().clone().unwrap();
        let rdata = parent.deps[1].data.borrow().clone().unwrap();
        let data = op(ldata, rdata);
        parent.set_data(data);
    }

    fn handle_bin_op(
        &self,
        other: &Var<T>,
        var_type: VariableType,
        backward_fn: BackwardFn<T>,
    ) -> Var<T> {
        Var {
            id: uuid::Uuid::new_v4(),
            data: Rc::new(RefCell::new(None)),
            deps: vec![Box::new(self.copy()), Box::new(other.copy())],
            evaluated: false,
            grad: Rc::new(RefCell::new(None)),
            is_leaf: false,
            requires_grad: self.requires_grad || other.requires_grad,
            var_type: var_type,
            backward_fn: Some(backward_fn),
        }
    }

    /// Evaluate computation graph and populate the data of intermediary variables
    pub fn eval(&mut self) {
        let sorted = Self::topological_sort(self, false);

        for var in sorted {
            match var.var_type {
                VariableType::Input => (),
                VariableType::OpAdd => Self::eval_bin_op(&var, |a, b| a + b),
                VariableType::OpSub => Self::eval_bin_op(&var, |a, b| a - b),
                VariableType::OpMul => Self::eval_bin_op(&var, |a, b| a * b),
            }
        }

        self.evaluated = true;
    }

    pub fn reset_grad(&mut self) {
        let sorted = Self::topological_sort(self, false);
        for v in sorted {
            *v.grad.borrow_mut() = None
        }
    }

    pub fn requires_grad(&mut self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, val: bool) {
        self.requires_grad = val;
    }

    pub fn data(&self) -> Option<T> {
        self.data.as_ref().borrow().clone()
    }

    pub fn grad(&self) -> Option<T> {
        self.grad.as_ref().borrow().clone()
    }

    pub fn set_grad(&self, grad: T) {
        *self.grad.borrow_mut() = Some(grad);
    }

    pub fn set_data(&self, data: T) {
        *self.data.borrow_mut() = Some(data);
    }

    pub fn deps(&self) -> &Deps<T> {
        &self.deps
    }
}

/// Basic arithmetic ops implementations
impl<'a, T: ArithmeticOps + HasGrad<T> + Debug> Var<T> {
    pub fn add(&self, other: &Var<T>) -> Var<T> {
        self.handle_bin_op(other, VariableType::OpAdd, add_backward)
    }

    pub fn sub(&self, other: &Var<T>) -> Var<T> {
        self.handle_bin_op(other, VariableType::OpSub, sub_backward)
    }

    pub fn mul(&self, other: &Var<T>) -> Var<T> {
        self.handle_bin_op(other, VariableType::OpMul, mul_backward)
    }
}

/// Reduce arithmetic implementations
impl<T: ArithmeticOps + HasGrad<T> + Debug> Var<T> {}
