use ndarray::{arr0, Array, Axis, CowArray, Dim, IxDynImpl};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::panic;

use crate::traits::{Dot, HasGrad, Reduce, Transpose};

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

impl<'a> Dot for NDArray<'a> {
    type Output = NDArray<'a>;
    fn dot(&self, other: Self) -> Self {
        let lhs = if self.0.ndim() == 2 {
            let shape = self.0.shape();
            self.0.clone().into_shape((shape[0], shape[1])).unwrap()
        } else {
            panic!("dot() is only defined for rank-2 tensors")
        };

        let rhs = if other.0.ndim() == 2 {
            let shape = other.0.shape();
            other.0.clone().into_shape((shape[0], shape[1])).unwrap()
        } else {
            panic!("dot() is only defined for rank-2 tensors")
        };
        Self(CowArray::from(lhs.dot(&rhs).into_dyn()))
    }
}

impl<'a> Reduce for NDArray<'a> {
    fn sum(&self) -> Self {
        let sum = arr0(self.0.sum());
        Self(CowArray::from(sum).into_dyn())
    }

    fn sum_axis(&self, axis: usize) -> Self {
        let sum = self.0.sum_axis(Axis(axis));
        Self(CowArray::from(sum).into_dyn())
    }
}

impl<'a>  Transpose for NDArray<'a> {
    fn t(&self) -> Self {
        let transposed = if self.0.ndim() == 2 {
            let tr = self.0.t();
            let shape = tr.shape();
            tr.clone().into_shape((shape[0], shape[1])).unwrap()
        } else {
            panic!("transpose() is only defined for rank-2 tensors")
        };
        Self(CowArray::from(transposed.into_owned().into_dyn()))
    }
}