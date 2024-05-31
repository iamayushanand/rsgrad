
pub struct Tensor {
    pub buffer: Vec<f32>,
    pub stride: Vec<u32>,
}

impl Tensor {
    pub fn new(data: Vec<f32> , shape: &[u32]) -> Tensor {
        let ndims: usize = shape.len();
        let mut stride: Vec<u32> = vec![1; ndims];
        for idx in (1..ndims).rev() {
            stride[idx-1] = stride[idx]*shape[idx];
        }
        Tensor {buffer: data, stride: stride}
    }

    pub fn constant_fill(constant: f32, shape: &[u32]) -> Tensor {
        let size: u32 = shape.iter().fold(1, |acc, &x| acc * x);
        let mut data: Vec<f32> = vec![constant; usize::try_from(size).unwrap()];
        let ndims: usize = shape.len();
        let mut stride: Vec<u32> = vec![1; ndims];
        for idx in (1..ndims).rev() {
            stride[idx-1] = stride[idx]*shape[idx];
        }
        Tensor {buffer: data, stride: stride}
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
}
