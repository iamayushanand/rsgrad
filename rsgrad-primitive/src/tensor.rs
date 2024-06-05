
use std::rc::Rc;
use crate::ops::Op;
use crate::ops::Mult;
use std::cell::RefCell;

pub struct Tensor {
    pub buffer: Vec<f32>,
    pub stride: Vec<u32>,
    pub grad: Option<Rc<RefCell<Tensor>>>,
    pub children: Vec<Rc<RefCell<Tensor>>>,
    pub op: Option<Op>
}

impl Tensor {
    pub fn new(data: Vec<f32> , shape: &[u32]) -> Tensor {
        let ndims: usize = shape.len();
        let mut stride: Vec<u32> = vec![1; ndims];
        for idx in (1..ndims).rev() {
            stride[idx-1] = stride[idx]*shape[idx];
        }
        Tensor {buffer: data, stride: stride, grad: None, children: Vec::new(), op: None}
    }

    pub fn constant_fill(constant: f32, shape: &[u32]) -> Tensor {
        let size: u32 = shape.iter().fold(1, |acc, &x| acc * x);
        let mut data: Vec<f32> = vec![constant; usize::try_from(size).unwrap()];
        let ndims: usize = shape.len();
        let mut stride: Vec<u32> = vec![1; ndims];
        for idx in (1..ndims).rev() {
            stride[idx-1] = stride[idx]*shape[idx];
        }
        Tensor {buffer: data, stride: stride, grad: None, children: Vec::new(), op: None}
    }

    pub fn shape(&self) -> Vec<u32> {
        let ndims = self.stride.len();
        let buffer_size = self.buffer.len();
        let mut shape: Vec<u32> = vec![1; ndims];
        shape[0] = u32::try_from(buffer_size).unwrap()/self.stride[0];
        for idx in 1..ndims {
            shape[idx] = self.stride[idx-1]/self.stride[idx];
        }
        shape
    }

    pub fn at(&mut self, index: &[u32]) -> &mut f32 {
        assert_eq!(index.len(), self.stride.len());
        let ndims: usize = index.len();
        let mut offset: usize = 0;
        for idx in 0..ndims {
            offset += usize::try_from(index[idx]*self.stride[idx]).unwrap();
        }
        &mut self.buffer[offset]
    }

    pub fn clone(&self) -> Tensor {
        Tensor {buffer: self.buffer.clone(), stride: self.stride.clone(), grad: self.grad.clone(), children: self.children.clone(), op: self.op.clone()}
    }

    pub fn backward(&mut self, gradient: Rc<RefCell<Tensor>>) {
        self.grad = Some(gradient);
        if self.op.is_none() {
            return ()
        }
        let grads: Vec<Tensor> = self.op.as_ref().unwrap().fetch_grad(&self.children);
        let nchild = grads.len();
        for idx in 0..nchild {
            let child_grad: Tensor = Mult::forward_nograd(self.grad.as_ref().unwrap().clone(), Rc::new(RefCell::new(grads[idx].clone())));
            self.children[idx].borrow_mut().backward(Rc::new(RefCell::new(child_grad)));
        }
        
    }
}
