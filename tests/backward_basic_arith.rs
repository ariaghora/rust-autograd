#[cfg(test)]
mod test_var_api_v2 {
    use rust_autograd::variable::Var;

    #[test]
    fn test_eval() {
        let x = Var::new(1.0);
        let mut z = x.add(&x).add(&x);
        let mut a = z.add(&z);

        assert!(z.val() == Some(3.0));
        assert!(z.val() == Some(3.0)); // call for second time
        assert!(a.val() == Some(6.0));
    }

    #[test]
    fn test_add_backward() {
        let mut x = Var::new(2.0);
        x.set_requires_grad(true);
        let y = Var::new(3.0);
        let mut z = x.add(&y);

        z.backward();

        assert!(z.requires_grad()); // when x requires grad, z must also require grad
        assert!(z.grad_wrt(&x) == Some(1.0));

        let mut z = x.add(&x); // z = 2x, so dz/dx=2
        z.backward();
        assert!(z.grad_wrt(&x) == Some(2.0));
    }

    #[test]
    fn test_mul_backward() {
        let mut x = Var::new(2.0);
        x.set_requires_grad(true);
        let y = Var::new(3.0);

        // z = x * y
        let mut z = x.mul(&y);
        z.backward();
        assert!(z.grad_wrt(&x) == Some(3.0)); // dz/dx == 3?

        // z = x^3 + y
        z = (x.mul(&x).mul(&x)).add(&y);
        z.backward();
        assert!(z.grad_wrt(&x) == Some(12.0)); // dz/dx == 12?
    }
}
