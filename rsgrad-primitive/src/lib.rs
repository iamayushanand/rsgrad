mod tensor;
mod ops;
use tensor::Tensor;

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
        let a = Tensor::constant_fill(1.0, shape);
        let b = Tensor::constant_fill(2.0, shape);
        let mut result = ops::Add::forward(&a, &b);
        assert_eq!(*result.at(&[1, 1, 1]), 3.0);
    }

    #[test]
    fn multiplication_test() {
        let shape: &[u32] = &[3, 2, 4];
        let a = Tensor::constant_fill(2.0, shape);
        let b = Tensor::constant_fill(3.0, shape);
        let mut result = ops::Mult::forward(&a, &b);
        assert_eq!(*result.at(&[1, 1, 1]), 6.0);
    }
}
