use std::{
    borrow::Borrow,
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    ops::Add,
    rc::Rc,
    vec,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum VariableType {
    Input,
    OpAdd,
}

pub trait HasGrad<T> {
    fn get_zero_grad(&self) -> Self;
    fn get_default_init_grad(&self) -> Self;
}

type ValueMap<T> = HashMap<uuid::Uuid, Rc<RefCell<Option<T>>>>;

fn add_backward<'a, T: HasGrad<T> + Add<Output = T> + Copy + Debug>(
    deps: &Vec<Box<Var<T>>>,
    data_map: &mut ValueMap<T>,
    grad_map: &mut ValueMap<T>,
    parent_grad: T,
) {
    let l_dep = deps.get(0).unwrap();
    let r_dep = deps.get(1).unwrap();

    if l_dep.requires_grad {
        let l_data = get_value(data_map.get(&l_dep.id).unwrap()).unwrap();
        let l_grad = match grad_map.get(&l_dep.id) {
            Some(grad_rc) => get_value(grad_rc).unwrap(),
            None => l_data.get_zero_grad(),
        };
        let new_grad = l_grad + parent_grad;
        grad_map.insert(l_dep.id, Rc::new(RefCell::new(Some(new_grad))));
    }

    if r_dep.requires_grad {
        let r_data = get_value(data_map.get(&r_dep.id).unwrap()).unwrap();
        let r_grad = match grad_map.get(&r_dep.id) {
            Some(grad_rc) => get_value(grad_rc).unwrap(),
            None => r_data.get_zero_grad(),
        };
        let new_grad = r_grad + parent_grad;
        grad_map.insert(r_dep.id, Rc::new(RefCell::new(Some(new_grad))));
    }
}

#[derive(Clone)]
pub struct Var<T> {
    id: uuid::Uuid,
    data: Rc<RefCell<Option<T>>>, // T is None for non-leaf nodes
    deps: Vec<Box<Var<T>>>,
    evaluated: bool,
    grad: Rc<Option<RefCell<T>>>,
    is_leaf: bool,
    requires_grad: bool,
    var_type: VariableType,

    /// Maintained by the root node and populated when during eval()
    data_map: Option<ValueMap<T>>,
    grad_map: Option<ValueMap<T>>,

    backward_fn: Option<fn(&Vec<Box<Var<T>>>, &mut ValueMap<T>, &mut ValueMap<T>, T)>,
}

impl<'a, T: HasGrad<T> + Copy + Add<Output = T> + Debug> Var<T> {
    pub fn new(data: T) -> Self {
        Var {
            id: uuid::Uuid::new_v4(),
            data: Rc::new(RefCell::new(Some(data))),
            deps: vec![],
            evaluated: false,
            grad: Rc::new(None),
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
            grad: if deep {
                self.grad.clone()
            } else {
                Rc::clone(&self.grad)
            },
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
        stack: &mut VecDeque<Var<T>>,
        allow_revisit: bool,
    ) {
        if !allow_revisit {
            if visited.contains(&variable.id) {
                return;
            }
        }

        visited.insert(variable.id);

        for dep in &variable.deps {
            self.dfs(dep, visited, stack, allow_revisit);
        }

        stack.push_back(variable.copy(false));
    }

    fn topological_sort(&self, entry: &Var<T>, allow_revisit: bool) -> Vec<Var<T>> {
        let mut visited = HashSet::new();
        let mut stack = VecDeque::new();

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

        for i in 0..sorted.len() {
            let var = &mut sorted[i];
            match var.backward_fn {
                Some(bw_fn) => {
                    // var requires grad. Proceed.
                    let deps = &var.deps;
                    let var_val = get_value(data_map.get(&var.id).unwrap()).unwrap();
                    let grad = var_val.get_default_init_grad();
                    bw_fn(deps, data_map, grad_map, grad);
                }
                None => (),
            }
        }
    }

    pub fn eval(&mut self) {
        self.data_map = Some(HashMap::new());

        let sorted = self.topological_sort(self, false);

        let data_map = &mut self.data_map.as_mut().unwrap();

        for var in sorted {
            match var.var_type {
                VariableType::Input => {
                    data_map.insert(var.id, Rc::clone(&var.data));
                }
                VariableType::OpAdd => {
                    // TODO: use function to handle binops
                    let ldata = data_map.get(&var.deps[0].id).unwrap();
                    let ldata = get_value(ldata).unwrap();
                    let rdata = data_map.get(&var.deps[1].id).unwrap();
                    let rdata = get_value(rdata).unwrap();
                    let data = ldata + rdata;
                    data_map.insert(var.id, Rc::new(RefCell::new(Some(data))));
                }
            }
        }

        self.evaluated = true;
    }

    fn get_from_data_map(&self, id: uuid::Uuid) -> T {
        let hm = self.data_map.as_ref().unwrap();
        let rc = hm.get(&id).borrow().unwrap();
        get_value(&rc).unwrap()
    }

    pub fn requires_grad(&mut self, val: bool) {
        self.requires_grad = val;
    }

    pub fn grad_wrt(&self, var: &Var<T>) -> T {
        let grad_map = self.grad_map.as_ref().unwrap();
        let grad = grad_map.get(&var.id).unwrap();
        get_value(grad).unwrap()
    }

    pub fn val(&mut self) -> T {
        if !self.evaluated {
            self.eval();
        }
        return self.get_from_data_map(self.id);
    }
}

/// Ops implementations
impl<'a, T: HasGrad<T> + Copy + Add<Output = T> + Debug> Var<T> {
    pub fn add(&self, other: &Var<T>) -> Var<T> {
        Var {
            id: uuid::Uuid::new_v4(),
            data: Rc::new(RefCell::new(None)),
            deps: vec![Box::new(self.copy(false)), Box::new(other.copy(false))],
            evaluated: false,
            grad: Rc::new(None),
            is_leaf: false,
            requires_grad: self.requires_grad || other.requires_grad,
            var_type: VariableType::OpAdd,
            data_map: None,
            grad_map: None,
            backward_fn: Some(add_backward),
        }
    }
}

fn get_value<T>(x: &Rc<RefCell<Option<T>>>) -> Option<T>
where
    T: Copy,
{
    let x_borrowed = x.as_ref().borrow();

    match &*x_borrowed {
        Some(value) => Some(*value),
        None => None,
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

#[test]
fn test_variable2_eval() {
    let x = Var::new(1.0);
    let mut z = x.add(&x).add(&x);
    let mut a = z.add(&z);

    assert!(z.val() == 3.0);
    assert!(z.val() == 3.0); // call for second time
    assert!(a.val() == 6.0);
}

#[test]
fn test_add_backward() {
    let mut x = Var::new(2.0);
    x.requires_grad(true);
    let y = Var::new(3.0);
    let mut z = x.add(&y);

    z.backward();

    assert!(z.requires_grad); // when x requires grad, z must also require grad
    assert!(z.grad_wrt(&x) == 1.0);

    let mut z = x.add(&x); // z = 2x, so dz/dx=2
    z.backward();
    assert!(z.grad_wrt(&x) == 2.0);
}
