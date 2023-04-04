use crate::variable::{VarType, Variable};
use paste::paste;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Add, Div, Mul, Neg, Sub};
use uuid::Uuid;

#[derive(Debug)]
pub struct Context<T> {
    data_map: HashMap<uuid::Uuid, T>,
    evaluated: bool,
}

macro_rules! make_trait {
    ($name:ident, $($trait:path),+ $(,)?) => {
        paste! {
            // Create the custom trait that combines the required traits
            pub trait $name: $($trait<Output = Self> +)+ Copy + Clone {}

            // Implement the custom trait for all types that satisfy the trait bounds
            impl<T: $($trait<Output = T> +)+ Copy + Clone> $name for T {}
        }
    };
}

make_trait!(ArithmeticOps, Add, Mul, Neg, Sub, Div);

impl<T: ArithmeticOps> Context<T> {
    pub fn new() -> Self {
        Context {
            data_map: HashMap::new(),
            evaluated: false,
        }
    }

    fn topological_sort_helper(
        &self,
        variable: &Variable,
        visited: &mut HashSet<Uuid>,
        stack: &mut VecDeque<Variable>,
        allow_revisit: bool,
    ) {
        if !allow_revisit {
            if visited.contains(&variable.data_id) {
                return;
            }
        }

        visited.insert(variable.data_id);

        for dep in &variable.deps {
            self.topological_sort_helper(dep, visited, stack, allow_revisit);
        }

        stack.push_back(variable.clone());
    }

    pub fn topological_sort(&self, entry: &Variable, allow_revisit: bool) -> Vec<Variable> {
        let mut visited = HashSet::new();
        let mut stack = VecDeque::new();

        self.topological_sort_helper(entry, &mut visited, &mut stack, allow_revisit);

        stack.into_iter().collect()
    }

    fn handle_binop(&mut self, var: &Variable, op: impl Fn(T, T) -> T) {
        let lval = self.data_map[&var.deps[0].data_id];
        let rval = self.data_map[&var.deps[1].data_id];
        self.data_map.insert(var.data_id, op(lval, rval));
    }

    pub fn eval(&mut self, root: &Variable) {
        let sorted = self.topological_sort(root, false);
        for var in sorted {
            match var.var_type {
                VarType::Leaf => (),
                VarType::OpAdd => self.handle_binop(&var, |a, b| a + b),
                VarType::OpSub => self.handle_binop(&var, |a, b| a - b),
                VarType::OpMul => self.handle_binop(&var, |a, b| a * b),
                VarType::OpDiv => self.handle_binop(&var, |a, b| a / b),
            }
        }
        self.evaluated = true;
    }

    pub fn value_of(&mut self, variable: &Variable) -> T {
        self.eval(&variable);
        let val = self.data_map.get(&variable.data_id);

        *val.unwrap()
    }

    pub fn var(&mut self, data: T) -> Variable {
        let var: Variable = Default::default();
        self.data_map.insert(var.data_id, data);

        var
    }
}

#[cfg(test)]
mod test {
    use super::Context;

    #[test]
    fn test_topo_sort() {
        let mut c = Context::new();
        let x = &c.var(1);
        let y = &c.var(2);
        let z = &x.add(y);
        let a = &z.add(z);

        // topo-sort without allowing revisiting, usually for evaluating
        // graph in forward pass
        let sorted = c.topological_sort(a, false);
        assert!(sorted[0].data_id == x.data_id);
        assert!(sorted[1].data_id == y.data_id);
        assert!(sorted[2].data_id == z.data_id);
        assert!(sorted[3].data_id == a.data_id);

        // topo-sort with allowing revisiting, usually for evaluating
        // graph in backward pass
        let sorted = c.topological_sort(a, true);
        assert!(sorted[0].data_id == x.data_id);
        assert!(sorted[1].data_id == y.data_id);
        assert!(sorted[2].data_id == z.data_id);
        assert!(sorted[3].data_id == x.data_id);
        assert!(sorted[4].data_id == y.data_id);
        assert!(sorted[5].data_id == z.data_id);
        assert!(sorted[6].data_id == a.data_id);
    }

    #[test]
    fn test_arith() {
        let mut c = Context::new();
        let x = c.var(4);
        let y = c.var(2);

        let z = &x.add(&y);
        assert!(c.value_of(z) == 6);

        let z = &x.sub(&y);
        assert!(c.value_of(z) == 2);

        let z = &x.mul(&y);
        assert!(c.value_of(z) == 8);

        let z = &x.div(&y);
        assert!(c.value_of(z) == 2);
    }
}
