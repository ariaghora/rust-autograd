use ndarray::{Array, CowArray, Dim, IxDynImpl};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::traits::HasGrad;

#[derive(Clone, Debug)]
pub struct NDArray<'a>(pub CowArray<'a, f64, Dim<IxDynImpl>>);

impl<'a> Display for NDArray<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'a> HasGrad<NDArray<'a>> for NDArray<'a> {
    fn get_zero_grad(&self) -> Self {
        NDArray(CowArray::from(Array::zeros(self.0.shape())))
    }

    fn get_default_init_grad(&self) -> Self {
        NDArray(CowArray::from(Array::ones(self.0.shape())))
    }
}

impl<'a> Add for NDArray<'a> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(rhs.0.into_owned()).into())
    }
}

impl<'a> Sub for NDArray<'a> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(rhs.0.into_owned()).into())
    }
}

impl<'a> Mul for NDArray<'a> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0.into_owned()).into())
    }
}

impl<'a> Div for NDArray<'a> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0.into_owned()).into())
    }
}

impl<'a> Neg for NDArray<'a> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(CowArray::from(self.0.neg()))
    }
}

#[cfg(test)]
mod test_ndarray {
    use crate::variable::from_ndarray;
    use ndarray::{array, Array};

    #[test]
    fn test_eval_ndarray() {
        let mut x = from_ndarray(Array::ones([2]));
        x.set_requires_grad(true);
        let mut y = x.add(&x).add(&x).add(&x);
        y.backward();

        let x_grad = x.grad().unwrap().0;
        let expected = array![4., 4.].into_dyn();
        assert!(x_grad.eq(&expected));
    }
}
