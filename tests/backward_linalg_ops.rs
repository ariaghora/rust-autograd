#[cfg(test)]
mod test_var_api_v2 {

    use ndarray::array;
    use rust_autograd::variable::from_ndarray;

    #[test]
    fn test_eval_dot() {
        let x = from_ndarray(array![[1., 1.], [2., 2.]]);
        let y = from_ndarray(array![[3.], [5.]]);
        let mut z = x.dot(&y);
        z.eval();

        let exp = array![[8.], [16.]];
        assert!(z.data().unwrap().0.eq(&exp.into_dyn()));
    }

    #[test]
    fn test_dot_backward() {
        let mut x = from_ndarray(array![[1., 1.], [2., 2.]]);
        let mut y = from_ndarray(array![[3.], [5.]]);
        x.set_requires_grad(true);
        y.set_requires_grad(true);
        let mut z = x.dot(&y);

        z.backward();

        let expected_x_grad = array![[3., 5.], [3., 5.]];
        let expected_y_grad = array![[3.], [3.]];
        assert!(x.grad().unwrap().0.eq(&expected_x_grad.into_dyn()));
        assert!(y.grad().unwrap().0.eq(&expected_y_grad.into_dyn()));
    }
}
