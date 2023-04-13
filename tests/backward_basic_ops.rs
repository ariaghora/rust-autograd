#[cfg(test)]
mod test_var_api_v2 {

    use ndarray::array;
    use rust_autograd::variable::from_ndarray;

    #[test]
    fn test_eval() {
        let x = from_ndarray(array![1.]);
        let mut z = x.add(&x).add(&x);
        let mut a = z.add(&z);

        z.eval();
        a.eval();

        assert!(z.data().unwrap().0.eq(&array![3.0].into_dyn()));
        assert!(a.data().unwrap().0.eq(&array![6.0].into_dyn()));
    }

    #[test]
    fn test_add_backward() {
        let mut x = from_ndarray(array![2., 2.]);
        x.set_requires_grad(true);
        let y = from_ndarray(array![3., 3.]);
        let mut z = x.add(&y);

        z.backward();
        assert!(z.requires_grad()); // when x requires grad, z must also require grad
        assert!(x.grad().unwrap().0.eq(&array![1., 1.].into_dyn()));

        z.reset_grad();

        z = x.add(&x);
        z.backward();
        assert!(x.grad().unwrap().0.eq(&array![2., 2.].into_dyn()));
    }

    #[test]
    fn test_mul_backward() {
        let mut x = from_ndarray(array![2.0]);
        x.set_requires_grad(true);
        let y = from_ndarray(array![3.0]);

        // z = x * y
        let mut z = x.mul(&y);
        z.backward();
        assert!(x.grad().unwrap().0 == &array![3.].into_dyn()); // dz/dx == 3?

        z.reset_grad();

        // z = x^3 + y
        z = (x.mul(&x).mul(&x)).add(&y);
        z.backward();
        assert!(x.grad().unwrap().0 == &array![12.].into_dyn()); // dz/dx == 3?
    }
}
