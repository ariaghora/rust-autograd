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
}
