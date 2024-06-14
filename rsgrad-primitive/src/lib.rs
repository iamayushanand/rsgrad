pub mod tensor;
pub mod ops;
use tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;
    //mod tensor;
    //use tensor::Tensor;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn tensor_initialisation() {
        let data: Vec<f32> = vec![1.0; 10];
        let shape: &[u32] = &[3, 2, 4];
        let a = Tensor::new(data, shape);
        assert_eq!(a.stride, vec![8, 4, 1]);
    }

    #[test]
    fn constant_tensor() {
        let shape: &[u32] = &[3, 2, 4];
        let a = Tensor::constant_fill(1.0, shape);
        assert_eq!(a.buffer, vec![1.0; 24]);
        assert_eq!(a.stride, vec![8, 4, 1]);
    }

    #[test]
    fn rand_tensor() {
        let shape: &[u32] = &[3, 2, 4];
        let a = Tensor::rand(shape);
        assert_eq!(a.stride, vec![8, 4, 1]);
    }

    #[test]
    fn tensor_at() {
        let shape: &[u32] = &[3, 2, 4];
        let mut a = Tensor::constant_fill(1.0, shape);
        assert_eq!(a.at(&[1, 1, 1]), &1.0);


        *a.at(&[1, 1, 1]) = 2.0;
        assert_eq!(a.at(&[1, 1, 1]), &2.0);
    }

    #[test]
    fn addition_test() {
        let shape: &[u32] = &[3, 2, 4];
        let a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let b = Rc::new(RefCell::new(Tensor::constant_fill(3.0, shape)));
        let mut result = ops::Add::forward(a, b);
        assert_eq!(*result.at(&[1, 1, 1]), 5.0);
    }

    #[test]
    fn multiplication_test() {
        let shape: &[u32] = &[3, 2, 4];
        let a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let b = Rc::new(RefCell::new(Tensor::constant_fill(3.0, shape)));
        let mut result = ops::Mult::forward(a, b);
        assert_eq!(*result.at(&[1, 1, 1]), 6.0);
    }

    #[test]
    fn addition_vjp_test() {
        let shape: &[u32] = &[3, 2, 4];
        let a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let b = Rc::new(RefCell::new(Tensor::constant_fill(3.0, shape)));
        //let mut result = ops::Add::forward(a, b);
        let mut x :Vec<Rc<RefCell<Tensor>>> = Vec::new();
        x.push(a.clone());
        x.push(b.clone());
        let grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, shape)));
        let mut result = ops::Add::vjp(grad, &x);
        assert_eq!(*result[0].at(&[1, 1, 1]), 1.0);
    }

    #[test]
    fn multiplication_vjp_test() {
        let shape: &[u32] = &[3, 2, 4];
        let a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let b = Rc::new(RefCell::new(Tensor::constant_fill(3.0, shape)));
        //let mut result = ops::Add::forward(a, b);
        let mut x :Vec<Rc<RefCell<Tensor>>> = Vec::new();
        x.push(a.clone());
        x.push(b.clone());
        let grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, shape)));
        let mut result = ops::Mult::vjp(grad, &x);
        assert_eq!(*result[0].at(&[1, 1, 1]), 3.0);
    }

    #[test]
    fn backward_test() {
        let shape: &[u32] = &[3, 2, 4];
        let mut a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let mut a_local = a.clone();
        let mut b = Rc::new(RefCell::new(Tensor::constant_fill(3.0, shape)));
        let mut b_local = b.clone();
        let mut c = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let mut c_local = c.clone();
        let mut intermediate = Rc::new(RefCell::new(ops::Add::forward(a, b)));
        let mut result = ops::Mult::forward(c, intermediate);
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, shape)));
        result.backward(init_grad);
        assert_eq!(*result.grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 1.0);
        assert_eq!(*a_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 2.0);
        assert_eq!(*b_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 2.0);
        assert_eq!(*c_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 5.0);
    }

    #[test]
    fn backward_test_with_log() {
        let shape: &[u32] = &[3, 2, 4];
        let mut a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let mut a_local = a.clone();
        let mut b = Rc::new(RefCell::new(Tensor::constant_fill(3.0, shape)));
        let mut b_local = b.clone();
        let mut c = Rc::new(RefCell::new(Tensor::constant_fill(2.0, shape)));
        let mut c_local = c.clone();
        let mut a_log = Rc::new(RefCell::new(ops::Log::forward(a)));
        let mut intermediate = Rc::new(RefCell::new(ops::Add::forward(a_log, b)));
        let mut result = ops::Mult::forward(c, intermediate);
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, shape)));
        result.backward(init_grad);
        assert_eq!(*result.grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 1.0);
        assert_eq!(*a_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 1.0);
        assert_eq!(*b_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 2.0);
        assert_eq!(*c_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1,1]), 3.0+(2.0_f32).ln());
    }

    #[test]
    fn matmul_test() {
        let mut a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, &[6, 2])));
        let mut a_local = a.clone();
        let mut b = Rc::new(RefCell::new(Tensor::constant_fill(3.0, &[2, 4])));
        let mut b_local = b.clone();
        let mut result = ops::MatMul::forward(a, b);
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[6, 4])));
        result.backward(init_grad);
        assert_eq!(*result.grad.as_ref().unwrap().borrow_mut().at(&[1,1]), 1.0);
        assert_eq!(*a_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1]), 12.0);
        assert_eq!(*b_local.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1,1]), 12.0);
    }

    #[test]
    fn L2norm_test() {
        let mut a = Rc::new(RefCell::new(Tensor::constant_fill(2.0, &[6])));
        let mut result = ops::L2norm::forward(a.clone());
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1])));
        result.backward(init_grad);
        assert_eq!(*result.at_im(&[0]), 24.0);
        assert_eq!(*a.borrow_mut().grad.as_ref().unwrap().borrow_mut().at(&[1]), 4.0);
    }

}
