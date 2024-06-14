use rsgrad_primitive::ops;
use rsgrad_primitive::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

pub struct SGD {
    params: Vec<Rc<RefCell<Tensor>>>,
    lr: f32
}

impl SGD {
    pub fn new(params: Vec<Rc<RefCell<Tensor>>>, lr: f32) -> SGD {
        SGD {params: params, lr: lr}
    }

    pub fn step(&self) {
        for param in &self.params {
            let size = param.borrow().buffer.len();
            for i in 0..size {
                let step_size = self.lr * param.borrow().grad.as_ref().unwrap().borrow().buffer[i];
                param.borrow_mut().buffer[i] -= step_size;
            }
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad=None;
        }
    }
}
