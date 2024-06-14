use rsgrad_primitive::tensor::Tensor;
use rsgrad_primitive::ops;
use std::cell::RefCell;
use std::rc::Rc;

pub struct Linear {
    pub param: Rc<RefCell<Tensor>>,
}

impl Linear {
    pub fn new(in_shape: u32, out_shape: u32) -> Linear {
        let mut param = Rc::new(RefCell::new(Tensor::rand(&[in_shape, out_shape])));
        Linear {param: param}
    }

    pub fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        Rc::new(RefCell::new(ops::MatMul::forward(x, self.param.clone())))
    }
}

pub struct ReLU;

impl ReLU {
    pub fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        Rc::new(RefCell::new(ops::Relu::forward(x)))
    }
}
