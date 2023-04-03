use crate::variable::{Op, Variable};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Add, Mul};
use uuid::Uuid;

#[derive(Debug)]
pub struct Context<T> {
    data_map: HashMap<uuid::Uuid, T>,
    evaluated: bool,
}

impl<T: Add<Output = T> + Mul<Output = T> + Copy + Clone> Context<T> {
    pub fn new() -> Self {
        Context {
            data_map: HashMap::new(),
            evaluated: false,
        }
    }

    fn dfs_topological_sort(
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
            self.dfs_topological_sort(dep, visited, stack, allow_revisit);
        }

        stack.push_back(variable.clone());
    }

    pub fn topological_sort_from_entry(
        &self,
        entry: &Variable,
        allow_revisit: bool,
    ) -> Vec<Variable> {
        let mut visited = HashSet::new();
        let mut stack = VecDeque::new();

        self.dfs_topological_sort(entry, &mut visited, &mut stack, allow_revisit);

        stack.into_iter().collect()
    }

    pub fn eval(&mut self, root: &Variable) {
        let sorted = self.topological_sort_from_entry(root, false);
        for var in sorted {
            match var.op {
                Op::NopScalar => (),
                Op::Add => {
                    let id = var.data_id;
                    let lval = self.data_map[&var.deps[0].data_id];
                    let rval = self.data_map[&var.deps[1].data_id];
                    let res = lval + rval;
                    self.data_map.insert(id, res);
                }
                Op::Mul => {
                    let id = var.data_id;
                    let lval = self.data_map[&var.deps[0].data_id];
                    let rval = self.data_map[&var.deps[1].data_id];
                    let res = lval * rval;
                    self.data_map.insert(id, res);
                }
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

        assert!(c.value_of(z) == 3);
        assert!(c.value_of(a) == 6);

        // topo-sort without allowing revisiting, usually for evaluating
        // expression in forward pass
        let sorted = c.topological_sort_from_entry(a, false);
        assert!(sorted[0].data_id == x.data_id);
        assert!(sorted[1].data_id == y.data_id);
        assert!(sorted[2].data_id == z.data_id);
        assert!(sorted[3].data_id == a.data_id);

        // topo-sort with allowing revisiting, usually for evaluating
        // expression in backward pass
        let sorted = c.topological_sort_from_entry(a, true);
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
        let x = c.var(1);
        let y = c.var(2);

        let z = &x.add(&y);
        assert!(c.value_of(z) == 3);
        let z = &x.mul(&y);
        assert!(c.value_of(z) == 2);
    }
}
