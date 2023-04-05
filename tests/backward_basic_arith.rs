#[cfg(test)]
mod test {

    use rust_autograd::context::Context;

    #[test]
    fn test_add_backward() {
        let mut c = Context::new();
        let mut x = c.var(1);
        x.requires_grad = true;
        let y = c.var(2);
        let z = &x + &y;
        c.backward(&z);

        assert!(z.requires_grad);
        assert!(c.grad_of(&x) == 1);

        let z = &x.add(&x).add(&x);
        c.backward(&z);
        assert!(c.grad_of(&x) == 3);
    }

    #[test]
    fn test_mul_backward() {
        let mut c = Context::new();
        let mut x = c.var(2);
        x.requires_grad = true;
        let y = c.var(3);
        let z = &x * &y;
        c.backward(&z);

        assert!(z.requires_grad);
        assert!(c.grad_of(&x) == 3);

        //     z = x^3
        // dz/dx = 3 * x^2
        //
        // when x = 2:
        // dz/dx = 3 * 2^2
        //       = 12
        let z = &x.mul(&x).mul(&x);
        c.backward(&z);
        assert!(c.grad_of(&x) == 12);
    }
}
