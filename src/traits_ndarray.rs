#[cfg(feature = "ndarray")]
use ndarray::{prelude::*, Array};

#[cfg(feature = "ndarray")]
#[test]
fn test_ndarray() {
    Array::from_vec(vec![1., 2., 3., 4.]);
}
