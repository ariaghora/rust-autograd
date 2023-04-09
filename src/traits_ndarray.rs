use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::traits::HasGrad;

#[derive(Clone, Debug)]
struct NDArray(ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>);

impl Display for NDArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl HasGrad<NDArray> for NDArray {
    fn get_zero_grad(&self) -> Self {
        // ArithmeticOps
        NDArray(Array::zeros(self.0.shape()))
    }

    fn get_default_init_grad(&self) -> Self {
        NDArray(Array::ones(self.0.shape()))
    }
}

impl Add for NDArray {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(rhs.0))
    }
}

impl Sub for NDArray {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(rhs.0))
    }
}

impl Mul for NDArray {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0))
    }
}

impl Div for NDArray {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0))
    }
}

impl Neg for NDArray {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

#[cfg(test)]
mod test_ndarray {
    use ndarray::Array;

    use crate::{traits_ndarray::NDArray, variable::Var};

    #[test]
    fn test_eval_ndarray() {
        let x = NDArray(Array::ones([2, 2]).into_dyn());

        let mut x = Var::new(x);
        x.set_requires_grad(true);
        let mut y = x.add(&x).add(&x).add(&x);
        y.backward();

        println!("{:?}", x.grad());
    }
}
