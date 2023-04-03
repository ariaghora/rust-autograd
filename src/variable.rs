use uuid::Uuid;

#[derive(Copy, Clone, Debug)]
pub enum Op {
    NopScalar,
    Add,
    Mul,
}

#[derive(Debug)]
pub struct Variable {
    pub(crate) data_id: Uuid,
    pub(crate) deps: Vec<Box<Variable>>,
    pub(crate) label: String,
    pub(crate) op: Op,
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        Self {
            data_id: self.data_id,
            deps: self.deps.clone(),
            label: self.label.clone(),
            op: self.op.clone(),
        }
    }
}

impl<'a, 'b> Variable {
    pub fn add(&'b self, other: &'b Variable) -> Variable {
        let mut new_var = Variable::default();
        new_var.op = Op::Add;
        new_var.deps.push(Box::new(self.clone()));
        new_var.deps.push(Box::new(other.clone()));
        new_var
    }

    pub fn mul(&'b self, other: &'b Variable) -> Variable {
        let mut new_var = Variable::default();
        new_var.op = Op::Mul;
        new_var.deps.push(Box::new(self.clone()));
        new_var.deps.push(Box::new(other.clone()));
        new_var
    }

    pub fn set_label(&mut self, label: String) {
        self.label = label;
    }

    pub fn data_id(&self) -> Uuid {
        self.data_id
    }
}

impl<'a> Default for Variable {
    fn default() -> Self {
        Self {
            data_id: Uuid::new_v4(),
            deps: Default::default(),
            label: Default::default(),
            op: Op::NopScalar,
        }
    }
}

impl std::ops::Add<&Variable> for &Variable {
    type Output = Variable;

    fn add(self, rhs: &Variable) -> Self::Output {
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
        let mut a: Variable = Default::default();
        let mut b: Variable = Default::default();
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
