use std::ops::{Add, Div, Mul, Neg, Sub};

use paste::paste;

macro_rules! make_arithmetic_ops_trait {
    ($name:ident, $($trait:path),+ $(,)?) => {
        paste! {
            // Create the custom trait that combines the required traits
            pub trait $name: $($trait<Output = Self> +)+ Copy  {}

            // Implement the custom trait for all types that satisfy the trait bounds
            impl<T: $($trait<Output = T> +)+ Copy> $name for T {}
        }
    };
}

make_arithmetic_ops_trait!(ArithmeticOps, Add, Mul, Neg, Sub, Div);

pub trait HasGrad<T> {
    fn get_zero_grad(&self) -> Self;
    fn get_default_init_grad(&self) -> Self;
}

pub trait GetSetById<T> {
    fn get_by_id(&self, id: uuid::Uuid) -> Option<T>;
    fn set_by_id(&mut self, id: uuid::Uuid, val: T);
}

impl HasGrad<f32> for f32 {
    fn get_zero_grad(&self) -> Self {
        0.0
    }

    fn get_default_init_grad(&self) -> Self {
        1.0
    }
}
