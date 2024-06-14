pub mod layer;
pub mod loss;
pub mod optimizer;
use optimizer::SGD;
use loss::L2loss;
use layer::Linear;
use layer::ReLU;
use rsgrad_primitive::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn linear_forward_test() {
        let linear_layer = Linear::new(3, 6);
        let x = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 3])));
        let res = linear_layer.forward(x);
        assert_eq!(linear_layer.param.borrow().shape(), &[3, 6]);
        assert_eq!(res.borrow().shape(), &[1, 6]);
    }

    #[test]
    fn linear_backward_test() {
        let linear_layer = Linear::new(3, 6);
        let x = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 3])));
        let mut res = linear_layer.forward(x.clone());
        assert_eq!(linear_layer.param.borrow().shape(), &[3, 6]);
        assert_eq!(res.borrow().shape(), &[1, 6]);
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 6])));
        res.borrow_mut().backward(init_grad);

        let mut weights = linear_layer.param.borrow_mut();
        assert_eq!(*(weights.grad.as_ref().unwrap().borrow_mut()).at(&[1, 1]), 1.0);
    }


    #[test]
    fn relu_backward_test() {
        let linear_layer = Linear::new(3, 6);
        let activation = ReLU;
        let x = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 3])));
        let mut res_1 = linear_layer.forward(x.clone());
        let mut res_2 = activation.forward(res_1.clone());
        assert_eq!(linear_layer.param.borrow().shape(), &[3, 6]);
        assert_eq!(res_2.borrow().shape(), &[1, 6]);
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 6])));
        res_2.borrow_mut().backward(init_grad);

        let mut weights = linear_layer.param.borrow_mut();
        assert_eq!(*(weights.grad.as_ref().unwrap().borrow()).at_im(&[1, 1]), 1.0);
    }

    #[test]
    fn relu_backward_test_negative() {
        let linear_layer = Linear::new(3, 6);
        let activation = ReLU;
        let x = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 3])));
        let mut res_1 = linear_layer.forward(x.clone());
        let mut res_2 = activation.forward(res_1.clone());
        assert_eq!(linear_layer.param.borrow().shape(), &[3, 6]);
        assert_eq!(res_2.borrow().shape(), &[1, 6]);
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 6])));
        res_2.borrow_mut().backward(init_grad);

        let mut weights = linear_layer.param.borrow_mut();
        assert_eq!(*(weights.grad.as_ref().unwrap().borrow()).at_im(&[1, 1]), 1.0);
    }

    #[test]
    fn L2loss_test() {
        let activation = ReLU;
        let x = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 3])));
        let t = Rc::new(RefCell::new(Tensor::constant_fill(3.0, &[1, 3])));
        let mut res_2 = activation.forward(x.clone());
        let mut res_3 = L2loss.forward(res_2.clone(), t);
        assert_eq!(res_3.buffer[0], 12.0);
    }

    #[test]
    fn SGD_test() {
        let activation = ReLU;
        let x = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1, 3])));
        let t = Rc::new(RefCell::new(Tensor::constant_fill(3.0, &[1, 3])));
        let mut res_2 = activation.forward(x.clone());
        let mut res_3 = L2loss.forward(res_2.clone(), t.clone());
        let mut params: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        params.push(res_2.clone());
        let mut optim = SGD::new(params, 0.01);
        let mut init_grad = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &[1])));
        res_3.backward(init_grad);
        assert_eq!(res_2.borrow().grad.as_ref().unwrap().borrow().buffer[0], -4.0);
        optim.step();
        assert_eq!(res_3.buffer[0], 12.0);
        assert_eq!(res_2.borrow().buffer[0], 1.04);
        optim.zero_grad();
    }
}
