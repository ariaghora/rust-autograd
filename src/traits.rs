use paste::paste;
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! make_arithmetic_ops_trait {
    ($name:ident, $($trait:path),+ $(,)?) => {
        paste! {
            // Create the custom trait that combines the required traits
            pub trait $name: $($trait<Output = Self> +)+ Copy + Clone {}

            // Implement the custom trait for all types that satisfy the trait bounds
            impl<T: $($trait<Output = T> +)+ Copy + Clone> $name for T {}
        }
    };
}

make_arithmetic_ops_trait!(ArithmeticOps, Add, Mul, Neg, Sub, Div);

pub trait GetGrad<T> {
    /// Get zero gradient value according to the value.
    /// For scalar, normally it returns zero. For ndarray, it will return
    /// ndarray of zeros with the same shape as the data itself.
    fn get_zero_grad(&self) -> T;

    /// Get initial downstream gradient when the node acts as root node.
    /// Usually one or array of ones.
    fn get_initial_grad(&self) -> T;
}

/// Helper when we want to use i32 as the data type
impl GetGrad<i32> for i32 {
    fn get_zero_grad(&self) -> i32 {
        0
    }
    fn get_initial_grad(&self) -> i32 {
        1
    }
}
