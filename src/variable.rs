use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::rc::Rc;
use std::vec;

use crate::traits::{ArithmeticOps, GetSetById, HasGrad};

#[derive(Clone, Copy, Debug, PartialEq)]
enum VariableType {
    Input,
    OpAdd,
    OpMul,
}

type BackwardFn<T> = fn(&Deps<T>, &mut ValueMap<T>, &mut ValueMap<T>, T);
type Deps<T> = Vec<Box<Var<T>>>;
type ValueMap<T> = HashMap<uuid::Uuid, Rc<RefCell<Option<T>>>>;

impl<T: Copy> GetSetById<T> for ValueMap<T> {
    fn get_by_id(&self, id: uuid::Uuid) -> Option<T> {
        match self.get(&id) {
            Some(v) => get_value(v),
            None => None,
        }
    }

    fn set_by_id(&mut self, id: uuid::Uuid, val: T) {
        self.insert(id, Rc::new(RefCell::new(Some(val))));
    }
}

fn get_value<T: Copy>(x: &Rc<RefCell<Option<T>>>) -> Option<T> {
    let x_borrowed = x.as_ref().borrow();
    match &*x_borrowed {
        Some(value) => Some(*value),
        None => None,
    }
}

fn add_backward<'a, T: HasGrad<T> + ArithmeticOps + Debug>(
    deps: &Vec<Box<Var<T>>>,
    data_map: &mut ValueMap<T>,
    grad_map: &mut ValueMap<T>,
    parent_grad: T,
) {
    let l_dep = &deps[0];
    let r_dep = &deps[1];

    if l_dep.requires_grad {
        let l_data = data_map.get_by_id(l_dep.id).unwrap();
        let l_grad = match grad_map.get_by_id(l_dep.id) {
            Some(grad) => grad,
            None => l_data.get_zero_grad(),
        };
        let new_grad = l_grad + parent_grad;
        grad_map.set_by_id(l_dep.id, new_grad);
    }

    if r_dep.requires_grad {
        let r_data = data_map.get_by_id(r_dep.id).unwrap();
        let r_grad = match grad_map.get_by_id(r_dep.id) {
            Some(grad) => grad,
            None => r_data.get_zero_grad(),
        };
        let new_grad = r_grad + parent_grad;
        grad_map.set_by_id(r_dep.id, new_grad);
    }
}

fn mul_backward<'a, T: HasGrad<T> + ArithmeticOps + Debug>(
    deps: &Deps<T>,
    data_map: &mut ValueMap<T>,
    grad_map: &mut ValueMap<T>,
    parent_grad: T,
) {
    let l_dep = &deps[0];
    let r_dep = &deps[1];

    if l_dep.requires_grad {
        let l_data = data_map.get_by_id(l_dep.id).unwrap();
        let r_data = data_map.get_by_id(r_dep.id).unwrap();

        let l_grad = match grad_map.get_by_id(l_dep.id) {
            Some(grad_rc) => grad_rc,
            None => l_data.get_zero_grad(),
        };
        let new_grad = l_grad + parent_grad * r_data;
        grad_map.set_by_id(l_dep.id, new_grad);
    }

    if r_dep.requires_grad {
        let r_data = data_map.get_by_id(r_dep.id).unwrap();
        let l_data = data_map.get_by_id(l_dep.id).unwrap();

        let r_grad = match grad_map.get_by_id(r_dep.id) {
            Some(grad_rc) => grad_rc,
            None => r_data.get_zero_grad(),
        };
        let new_grad = r_grad + parent_grad * l_data;
        grad_map.set_by_id(r_dep.id, new_grad);
    }
}

#[derive(Clone)]
pub struct Var<T> {
    id: uuid::Uuid,

    data: Rc<RefCell<Option<T>>>, // T is None for non-leaf nodes
    deps: Deps<T>,
    evaluated: bool,
    is_leaf: bool,
    requires_grad: bool,
    var_type: VariableType,

    /// Maintained by the root node and populated when during eval()
    data_map: Option<ValueMap<T>>,
    grad_map: Option<ValueMap<T>>,

    backward_fn: Option<BackwardFn<T>>,
}

impl<'a, T: HasGrad<T> + ArithmeticOps + Debug> Var<T> {
    pub fn new(data: T) -> Self {
        Var {
            id: uuid::Uuid::new_v4(),
            data: Rc::new(RefCell::new(Some(data))),
            deps: vec![],
            evaluated: false,
            is_leaf: true,
            requires_grad: false,
            var_type: VariableType::Input,
            data_map: None,
            grad_map: None,
            backward_fn: None,
        }
    }

    /// Make a cheap partial clone of self.
    fn copy(&self, deep: bool) -> Self {
        Self {
            id: self.id,
            data: if deep {
                self.data.clone()
            } else {
                Rc::clone(&self.data)
            },
            deps: self.deps.clone(),
            evaluated: self.evaluated,
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            var_type: self.var_type,

            // cloning map only occurs during topo-sort, and is fine since they'll still
            // be None. They'll be initiated during backward.
            data_map: self.data_map.clone(),
            grad_map: self.data_map.clone(),

            backward_fn: self.backward_fn.clone(),
        }
    }

    fn dfs(
        &self,
        variable: &Var<T>,
        visited: &mut HashSet<uuid::Uuid>,
        stack: &mut Vec<Var<T>>,
        allow_revisit: bool,
    ) {
        if allow_revisit || !visited.contains(&variable.id) {
            visited.insert(variable.id);
            for dep in &variable.deps {
                self.dfs(dep, visited, stack, allow_revisit);
            }
            stack.push(variable.copy(false));
        }
    }

    fn topological_sort(&self, entry: &Var<T>, allow_revisit: bool) -> Vec<Var<T>> {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        self.dfs(entry, &mut visited, &mut stack, allow_revisit);
        stack.into_iter().collect()
    }

    pub fn backward(&mut self) {
        self.eval();

        // backward requires reverse-topological sort, allowing revisiting
        let mut sorted = self.topological_sort(self, true);
        sorted.reverse();

        let (data_map, grad_map) = {
            let data_map = (self.data_map).as_mut().unwrap();

            self.grad_map = Some(HashMap::new());
            let grad_map = self.grad_map.as_mut().unwrap();
            (data_map, grad_map)
        };

        // run backward propagation in iterative manner
        for i in 0..sorted.len() {
            let var = &mut sorted[i];
            match var.backward_fn {
                Some(bw_fn) => {
                    // var requires grad. Proceed.
                    let deps = &var.deps;
                    let var_val = data_map.get_by_id(var.id).unwrap();

                    // The grad of root node is set from get_default_init_grad(), which is
                    // usually ones. Otherwise, get the grad from the grad_map by that node's id
                    let grad = if var.id == self.id {
                        var_val.get_default_init_grad()
                    } else {
                        grad_map.get_by_id(var.id).unwrap()
                    };

                    bw_fn(deps, data_map, grad_map, grad);
                }
                None => (),
            }
        }
    }

    pub fn eval_bin_op(parent: &Var<T>, data_map: &mut ValueMap<T>, op: fn(T, T) -> T) {
        let ldata = data_map.get_by_id(parent.deps[0].id).unwrap();
        let rdata = data_map.get_by_id(parent.deps[1].id).unwrap();
        let data = op(ldata, rdata);
        data_map.set_by_id(parent.id, data);
    }

    pub fn eval(&mut self) {
        let sorted = self.topological_sort(self, false);

        self.data_map = Some(HashMap::new());
        let data_map = &mut self.data_map.as_mut().unwrap();

        for var in sorted {
            match var.var_type {
                VariableType::Input => {
                    data_map.insert(var.id, Rc::clone(&var.data));
                }
                VariableType::OpAdd => Self::eval_bin_op(&var, data_map, |a, b| a + b),
                VariableType::OpMul => Self::eval_bin_op(&var, data_map, |a, b| a * b),
            }
        }

        self.evaluated = true;
    }

    pub fn requires_grad(&mut self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, val: bool) {
        self.requires_grad = val;
    }

    pub fn grad_wrt(&self, var: &Var<T>) -> Option<T> {
        match self.grad_map.as_ref() {
            Some(map) => map.get_by_id(var.id),
            None => None,
        }
    }

    pub fn val(&mut self) -> Option<T> {
        if !self.evaluated {
            self.eval();
        }
        match self.data_map.as_ref() {
            Some(map) => map.get_by_id(self.id),
            None => None,
        }
    }
}

/// Ops implementations
impl<'a, T: ArithmeticOps + HasGrad<T> + Debug> Var<T> {
    fn bin_op(&self, other: &Var<T>, var_type: VariableType, backward_fn: BackwardFn<T>) -> Var<T> {
        Var {
            id: uuid::Uuid::new_v4(),
            data: Rc::new(RefCell::new(None)),
            deps: vec![Box::new(self.copy(false)), Box::new(other.copy(false))],
            evaluated: false,
            is_leaf: false,
            requires_grad: self.requires_grad || other.requires_grad,
            var_type: var_type,
            data_map: None,
            grad_map: None,
            backward_fn: Some(backward_fn),
        }
    }

    pub fn add(&self, other: &Var<T>) -> Var<T> {
        self.bin_op(other, VariableType::OpAdd, add_backward)
    }

    pub fn mul(&self, other: &Var<T>) -> Var<T> {
        self.bin_op(other, VariableType::OpMul, mul_backward)
    }
}

impl HasGrad<f32> for f32 {
    fn get_zero_grad(&self) -> Self {
        0.0
    }

    fn get_default_init_grad(&self) -> Self {
        1.0
    }
}
