use crate::tensor::Tensor;

pub struct Add;

impl Add {
    pub fn forward(a: &Tensor, b: &Tensor)-> Tensor {
        let bufsize = (*a).buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = (*a).buffer[i]+(*b).buffer[i];
        }
        Tensor{buffer: buffer, stride: (*a).stride.clone()}
    }
}

pub struct Mult;

impl Mult {
    pub fn forward(a: &Tensor, b: &Tensor)-> Tensor {
        let bufsize = (*a).buffer.len();
        let mut buffer: Vec<f32> = vec![1.0; bufsize];
        for i in 0..bufsize {
            buffer[i] = (*a).buffer[i]*(*b).buffer[i];
        }
        Tensor{buffer: buffer, stride: (*a).stride.clone()}
    }
}