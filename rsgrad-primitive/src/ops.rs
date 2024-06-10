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

    pub fn grad(x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        let mut result: Vec<Tensor> = Vec::new();
        let mut ones_tensor = Tensor::constant_fill(1.0, &(x[0].borrow().shape())[..]);
        let mut ones_tensor_dup = Tensor::constant_fill(1.0, &(x[0].borrow().shape())[..]);
        result.push(ones_tensor);
        result.push(ones_tensor_dup);
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

    pub fn grad(x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        let mut result: Vec<Tensor> = Vec::new();
        result.push(x[1].borrow().clone());
        result.push(x[0].borrow().clone());
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

    pub fn grad(x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        let mut result: Vec<Tensor> = Vec::new();
        let a_tensor = x[0].borrow();
        let bufsize = a_tensor.buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = 1.0/a_tensor.buffer[i];
        }
        let res_tensor = Tensor{buffer: buffer, stride: a_tensor.stride.clone(), grad: None, children: Vec::new(), op: None};
        result.push(res_tensor);
        result
    }
}

#[derive(Clone)]
pub enum Op {
    ADD,
    MULT,
    LOG
}

impl Op {
    
    pub fn fetch_grad(&self, x: &Vec<Rc<RefCell<Tensor>>>) -> Vec<Tensor> {
        match self {
            Op::ADD => Add::grad(x),
            Op::MULT => Mult::grad(x),
            Op::LOG => Log::grad(x)
        }
    }
}