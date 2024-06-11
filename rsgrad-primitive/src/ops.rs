use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
pub struct Add;

impl Add {
    pub fn forward(a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>)-> Tensor {
        let a_tensor = a.borrow();
        let b_tensor = b.borrow();
        let bufsize = a_tensor.buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = a_tensor.buffer[i]+b_tensor.buffer[i];
        } 
        let mut children: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        children.push(a.clone());
        children.push(b.clone());
        Tensor{buffer: buffer, stride: a_tensor.stride.clone(), grad: None, children: children, op: Some(Op::ADD)}
        
    }

    pub fn vjp(grad: Rc<RefCell<Tensor>>, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        let mut result: Vec<Tensor> = Vec::new();
        let mut ones_tensor = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &(x[0].borrow().shape())[..])));
        let mut ones_tensor_dup = Rc::new(RefCell::new(Tensor::constant_fill(1.0, &(x[0].borrow().shape())[..])));
        
        let mut ones_tensor_rc = Mult::forward(grad.clone(), ones_tensor);
        let mut ones_tensor_rc_dup = Mult::forward(grad.clone(), ones_tensor_dup);
        result.push(ones_tensor_rc);
        result.push(ones_tensor_rc_dup);
        result
    }
}

#[derive(Clone)]
pub struct Mult;

impl Mult {
    pub fn forward_nograd(a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>)-> Tensor {
        let a_tensor = a.borrow();
        let b_tensor = b.borrow();
        let bufsize = a_tensor.buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = a_tensor.buffer[i]*b_tensor.buffer[i];
        } 
        Tensor{buffer: buffer, stride: a_tensor.stride.clone(), grad: None, children: Vec::new(), op: None}
    }

    pub fn forward(a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>)-> Tensor {
        let a_tensor = a.borrow();
        let b_tensor = b.borrow();
        let bufsize = a_tensor.buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = a_tensor.buffer[i]*b_tensor.buffer[i];
        } 
        let mut children: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        children.push(a.clone());
        children.push(b.clone());
        Tensor{buffer: buffer, stride: a_tensor.stride.clone(), grad: None, children: children, op: Some(Op::MULT)}
    }

    pub fn vjp(grad: Rc<RefCell<Tensor>>, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        let mut result: Vec<Tensor> = Vec::new();
        result.push(Mult::forward_nograd(grad.clone(), x[1].clone()));
        result.push(Mult::forward_nograd(grad.clone(), x[0].clone()));
        //result.push(x[0].borrow().clone());
        result
    }
}

#[derive(Clone)]
pub struct Log;

impl Log {

    pub fn forward(a: Rc<RefCell<Tensor>>)-> Tensor {
        let a_tensor = a.borrow();
        let bufsize = a_tensor.buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = a_tensor.buffer[i].ln();
        } 
        let mut children: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        children.push(a.clone());
        Tensor{buffer: buffer, stride: a_tensor.stride.clone(), grad: None, children: children, op: Some(Op::LOG)}
    }

    pub fn vjp(grad: Rc<RefCell<Tensor>>, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        let mut result: Vec<Tensor> = Vec::new();
        let a_tensor = x[0].borrow();
        let bufsize = a_tensor.buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = 1.0/a_tensor.buffer[i];
        }
        let res_tensor = Tensor{buffer: buffer, stride: a_tensor.stride.clone(), grad: None, children: Vec::new(), op: None};
        result.push(Mult::forward_nograd(grad, Rc::new(RefCell::new(res_tensor))));
        result
    }
}

#[derive(Clone)]
pub struct MatMul;

impl MatMul {

    
    pub fn forward_nograd(a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>)-> Tensor {
        let a_tensor = a.borrow();
        let b_tensor = b.borrow();
        assert_eq!(a_tensor.stride.len(), 2);
        assert_eq!(b_tensor.stride.len(), 2);

        let a_shape = a_tensor.shape();
        let b_shape = b_tensor.shape();
        assert_eq!(a_shape[1], b_shape[0]);
        let bufsize = a_shape[0]*b_shape[1];
        let m = a_shape[1];
        let mut buffer: Vec<f32> = vec![1.0; usize::try_from(bufsize).unwrap()];
        let mut result = Tensor::new(buffer, &[a_shape[0], b_shape[1]]);
        for i in 0..a_shape[0] {
            for j in 0..b_shape[1] {
                let mut current: f32 = 0.0;
                for k in 0..m {
                    current += (*a_tensor.at_im(&[i, k])) * (*b_tensor.at_im(&[k, j]));
                }
                *result.at(&[i, j]) = current;
            }
        }
        let mut children: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        children.push(a.clone());
        children.push(b.clone());
        result.children = children;
        result.op = Some(Op::MATMUL);
        result
    }
    
    
    pub fn forward(a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>)-> Tensor {
        let a_tensor = a.borrow();
        let b_tensor = b.borrow();
        assert_eq!(a_tensor.stride.len(), 2);
        assert_eq!(b_tensor.stride.len(), 2);

        let a_shape = a_tensor.shape();
        let b_shape = b_tensor.shape();
        let bufsize = a_shape[0]*b_shape[1];
        let m = a_shape[1];
        let mut buffer: Vec<f32> = vec![1.0; usize::try_from(bufsize).unwrap()];
        let mut result = Tensor::new(buffer, &[a_shape[0], b_shape[1]]);
        for i in 0..a_shape[0] {
            for j in 0..b_shape[1] {
                let mut current: f32 = 0.0;
                for k in 0..m {
                    current += (*a_tensor.at_im(&[i, k])) * (*b_tensor.at_im(&[k, j]));
                }
                *result.at(&[i, j]) = current;
            }
        }
        let mut children: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        children.push(a.clone());
        children.push(b.clone());
        result.children = children;
        result.op = Some(Op::MATMUL);
        result
    }

    pub fn vjp(grad: Rc<RefCell<Tensor>>, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        let mut result: Vec<Tensor> = Vec::new();
        
        let a_transpose = Rc::new(RefCell::new(x[0].borrow().transpose()));
        let b_transpose = Rc::new(RefCell::new(x[1].borrow().transpose()));

        result.push(MatMul::forward_nograd(grad.clone(), b_transpose));
        result.push(MatMul::forward_nograd(a_transpose, grad.clone()));
        result
    }
}

#[derive(Clone)]
pub enum Op {
    ADD,
    MULT,
    LOG,
    MATMUL
}

impl Op {
    
    pub fn fetch_vjp(&self, grad: Rc<RefCell<Tensor>>, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        match self {
            Op::ADD => Add::vjp(grad, x),
            Op::MULT => Mult::vjp(grad, x),
            Op::LOG => Log::vjp(grad, x),
            Op::MATMUL => MatMul::vjp(grad, x)
        }
    }
}