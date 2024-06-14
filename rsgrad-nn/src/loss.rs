use rsgrad_primitive::ops;
use rsgrad_primitive::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

pub struct L2loss;

impl L2loss {
    pub fn forward(&self, x: Rc<RefCell<Tensor>>, t: Rc<RefCell<Tensor>>) -> Tensor {
        let diff = Rc::new(RefCell::new(ops::Sub::forward(x, t)));
        let result = ops::L2norm::forward(diff);
        result
    }
}
